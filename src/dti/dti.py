"""
src/dti/dti.py

Deficit-Triggered Integration (DTI) — paper Section 6 + Appendix C.

DTI is a cascade-local intervention that monitors the imbalance between
expansion and integration within each active coordination cascade and
triggers a forced merge when the deficit exceeds a condition-specific
threshold.

Algorithm (paper Algorithm 1 + Appendix C, Eq. A1–A5):
─────────────────────────────────────────────────────────────────────
For each active root claim r, maintain local cascade state:
  t_r  — coordination events elapsed in current cascade segment
  M_r  — realized merge events in current cascade segment

Per event:
  1. Update t_r += 1; if MERGE, M_r += 1
  2. Compute exploration pressure:   P_r = a_c * t_r^β̂_c
  3. Compute integration deficit:    Δ_r = P_r - M_r
  4. If Δ_r > δ_c:
       B_r = active branch heads (most recent outputs causally attached to r)
       Invoke merge integration over B_r → produce merged claim ẽ
       Log ẽ as MERGE event attached to r
       Broadcast ẽ as updated shared context
       Restart:  t_r = 0, M_r = 1

Parameters (estimated from baseline traces, per condition class c):
  β̂_c  contradiction scaling exponent  (how fast pressure grows)
  a_c   normalization constant           (empirical pressure scale)
  δ_c   deficit threshold                (mean + 1σ of deficit at
                                          cascade termination points)

Condition class: topology × task_family  (e.g. "chain_reasoning")

Design notes:
─────────────────────────────────────────────────────────────────────
- DTI maintains O(|R|) memory where |R| = number of active root claims.
- Each event incurs O(1) updates to (t_r, M_r).
- No additional model calls except when deficit threshold is exceeded.
- Parameters are fixed from baseline logs before intervention and contain
  no outcome-tuned quantities (paper Appendix C).
- DTI does NOT alter agent prompts or capabilities. It intercepts the
  event stream between agent turns and injects a merge event when triggered.
- The integration prompt consolidates active branch positions, identifies
  agreement, and resolves disagreement. Content shaping is left to the LLM.

Where it lives in the codebase:
─────────────────────────────────────────────────────────────────────
DTI sits in dti/ alongside a future dti_sweep.py. It is instantiated
once per run in graph_runner.run() and receives every TraceRow via
observe() immediately after bus.log() writes it. If the deficit
threshold is exceeded, observe() calls the LLM, writes the merge
TraceRow to the same bus, and returns the row. Otherwise returns None.

Integration into graph_runner.py:
─────────────────────────────────────────────────────────────────────
    dti = DTIMonitor(
        llm=llm, bus=bus, run_id=run_id,
        params=DTIParams.for_condition(topology, task_family),
        task_id=task_id, benchmark=benchmark,
        task_family=task_family, topology=topology,
        num_agents=num_agents, seed=seed,
    )
    dti.set_root_task(task.prompt)

    # After each topo.run(node_prompt) call, read new rows and feed to DTI:
    for row in new_rows_since_last_topo_run:
        merge_row = dti.observe(row)
        # merge_row is None if no trigger, else already logged to bus

    dti.log_stats(run_dir)

Parameter estimation (run once after baseline sweep, before DTI sweep):
─────────────────────────────────────────────────────────────────────
    params_dict = estimate_dti_params_from_sweep(
        sweep_data_root=Path("data/sweep"),
        save_path=Path("data/dti_params.json"),
    )
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Parameter container ───────────────────────────────────────────────────────

@dataclass
class DTIParams:
    """
    Condition-specific DTI parameters (paper Appendix C).

    All three are estimated from baseline coordination traces before
    intervention and remain fixed during the run — no outcome-tuned
    quantities.

    condition_key:  "{topology}_{task_family}"  e.g. "chain_reasoning"
    beta_c:         contradiction scaling exponent (β̂_c ≈ 0.60, paper)
    a_c:            normalization constant (empirical pressure scale)
    delta_c:        deficit threshold = mean + 1σ of deficit at
                    cascade termination points
    """
    condition_key: str
    beta_c:        float = 0.60
    a_c:           float = 1.0
    delta_c:       float = 2.0

    @classmethod
    def from_baseline_traces(
        cls,
        condition_key: str,
        run_dirs: List[Path],
        beta_c: float = 0.60,
    ) -> "DTIParams":
        """
        Estimate a_c and δ_c from completed baseline (non-DTI) run directories.

        a_c estimation:
            Fit P_r = a_c * t^β̂_c to observed (t_r, M_r) pairs at merge
            events across all cascades in the run set. OLS on log scale:
            log(M_r) = log(a_c) + β̂_c * log(t_r)
            → log(a_c) = mean(log(M_r) - β̂_c * log(t_r))

        δ_c estimation (paper Appendix C):
            "δ_c is defined as the mean plus one standard deviation of
            the integration deficit observed at cascade termination points."
            Deficit at termination = P_r(t_r_final) - M_r_final
            where P_r uses the estimated a_c.
        """
        import numpy as np
        from event_extraction.event_extractor import annotate_event_types
        from observables.dag_builder import build_all

        merge_t: list[float] = []
        merge_m: list[float] = []
        termination_deficits: list[float] = []

        for run_dir in run_dirs:
            ef = run_dir / "events.jsonl"
            if not ef.exists():
                continue
            rows = [json.loads(l) for l in ef.read_text().splitlines() if l.strip()]
            if not rows:
                continue

            annotated = annotate_event_types(rows)
            _, _, cascades = build_all(annotated)

            for cascade in cascades:
                if cascade.tce < 3:   # skip trivially short cascades
                    continue

                # Replay cascade in step order to collect (t_r, M_r) trajectory
                cascade_rows = sorted(
                    [r for r in annotated
                     if r.get("root_claim_id") == cascade.root_claim_id],
                    key=lambda r: (r.get("step_id", 0), r.get("timestamp", 0.0)),
                )

                t_r, m_r = 0, 0
                for row in cascade_rows:
                    t_r += 1
                    if row.get("event_type") == "merge_claims":
                        m_r += 1
                        if t_r > 0 and m_r > 0:
                            merge_t.append(float(t_r))
                            merge_m.append(float(m_r))

                # Record deficit at termination (a_c=1 placeholder; rescaled below)
                if t_r > 0:
                    termination_deficits.append(float(t_r ** beta_c) - m_r)

        # Estimate a_c via log-linear OLS
        a_c = 1.0
        if len(merge_t) >= 5:
            log_t = np.log(np.array(merge_t))
            log_m = np.log(np.maximum(np.array(merge_m), 1e-6))
            log_a = float(np.mean(log_m - beta_c * log_t))
            a_c   = max(0.01, math.exp(log_a))

        # Estimate δ_c = mean + 1σ of termination deficits, scaled by a_c
        delta_c = 2.0
        if len(termination_deficits) >= 5:
            scaled  = [a_c * d for d in termination_deficits]
            delta_c = max(0.5, float(np.mean(scaled) + np.std(scaled)))

        return cls(
            condition_key=condition_key,
            beta_c=beta_c,
            a_c=a_c,
            delta_c=delta_c,
        )

    @classmethod
    def for_condition(
        cls,
        topology:    str,
        task_family: str,
        beta_c:      float = 0.60,
        a_c:         float = 1.0,
        delta_c:     float = 2.0,
    ) -> "DTIParams":
        """
        Construct params directly when baseline estimation has not been run yet.
        Use paper default β̂_c = 0.60; a_c and δ_c default to conservative values.
        """
        return cls(
            condition_key=f"{topology}_{task_family}",
            beta_c=beta_c,
            a_c=a_c,
            delta_c=delta_c,
        )

    def to_dict(self) -> dict:
        return {
            "condition_key": self.condition_key,
            "beta_c":        self.beta_c,
            "a_c":           self.a_c,
            "delta_c":       self.delta_c,
        }


# ── Per-cascade state ─────────────────────────────────────────────────────────

@dataclass
class _CascadeState:
    """
    Local state for one active root claim.  O(1) per event update.
    Corresponds to (t_r, M_r) in paper Algorithm 1.
    """
    root_claim_id: str
    t_r:           int = 0    # coordination events in current segment
    m_r:           int = 0    # merge events in current segment
    trigger_count: int = 0    # how many times DTI has fired on this cascade

    # branch_heads: agent_id → most recent TraceRow for this cascade
    # Updated every event; B_r = values of this dict at trigger time
    branch_heads:  Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ── Integration prompt ────────────────────────────────────────────────────────

_DTI_SYSTEM_PROMPT = (
    "You are a synthesis agent. You will receive the most recent reasoning "
    "outputs from multiple active branches in an ongoing multi-agent "
    "coordination cascade. Your task is to consolidate these into a single "
    "merged claim that:\n"
    "  1. Identifies points of agreement across branches.\n"
    "  2. Resolves remaining disagreements by choosing the best-supported "
    "position or synthesising a reconciled view.\n"
    "  3. Produces a unified, self-contained output that advances the "
    "collective reasoning without losing important content from any branch.\n"
    "Return valid JSON only."
)


def _build_integration_prompt(
    branch_outputs: List[Dict[str, Any]],
    root_task: str,
) -> str:
    """
    Build the user prompt for the DTI integration call.
    B_r = active branch heads causally attached to root claim r.
    """
    lines = [f"Root task:\n{root_task}\n\n"
             f"Active branch outputs ({len(branch_outputs)} branches):"]
    for i, b in enumerate(branch_outputs):
        agent_id  = b.get("agent_id", f"agent_{i}")
        claim_id  = b.get("claim_id", "unknown")
        text      = b.get("reasoning_text") or b.get("answer", "")
        if len(text) > 600:
            text = text[:597] + "..."
        lines.append(
            f"\n[Branch {i+1}]  agent={agent_id}  claim_id={claim_id}\n{text}"
        )
    lines.append(
        '\n\nConsolidate the above into one merged claim. '
        'Return JSON: {"answer": "...", "reasoning": "...", "confidence": 0.0}'
    )
    return "\n".join(lines)


# ── DTI Monitor ───────────────────────────────────────────────────────────────

class DTIMonitor:
    """
    Cascade-local DTI monitor (paper Algorithm 1).

    Receives every TraceRow via observe() immediately after it is logged.
    Maintains per-root-claim state (t_r, M_r). Triggers a merge integration
    call when the integration deficit Δ_r = P_r - M_r exceeds δ_c.

    The triggered merge is logged to the same EventBus as all other events,
    so it appears in events.jsonl as a normal merge_claims TraceRow and is
    picked up by the post-hoc pipeline (event_extractor → dag_builder →
    cascade_metrics) without any special handling.
    """

    def __init__(
        self,
        llm:         Any,
        bus:         Any,          # EventBus
        run_id:      str,
        params:      DTIParams,
        task_id:     str = "",
        benchmark:   str = "",
        task_family: str = "",
        topology:    str = "",
        num_agents:  int = 0,
        seed:        int = 0,
    ) -> None:
        self.llm         = llm
        self.bus         = bus
        self.run_id      = run_id
        self.params      = params
        self.task_id     = task_id
        self.benchmark   = benchmark
        self.task_family = task_family
        self.topology    = topology
        self.num_agents  = num_agents
        self.seed        = seed

        self._cascades:      Dict[str, _CascadeState] = {}
        self._root_task:     str  = ""
        self._n_triggers:    int  = 0
        self._n_events_seen: int  = 0
        self._trigger_log:   list = []

    def set_root_task(self, task_text: str) -> None:
        """Provide the root task description for integration prompts."""
        self._root_task = task_text

    def observe(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Feed one TraceRow to DTI.

        Returns the merge TraceRow if DTI triggered (already logged to bus),
        else None.
        """
        self._n_events_seen += 1

        et = row.get("event_type", "")
        if et not in ("propose_claim", "revise_claim", "contradict_claim",
                      "merge_claims", "endorse_claim", "delegate_subtask"):
            return None

        root_id = row.get("root_claim_id") or row.get("claim_id")
        if not root_id:
            return None

        state = self._cascades.setdefault(
            root_id, _CascadeState(root_claim_id=root_id)
        )

        # Track most recent output per agent for this cascade (B_r candidates)
        agent_id = row.get("agent_id")
        if agent_id:
            state.branch_heads[agent_id] = row

        # Algorithm 1, lines 6-9: update t_r and M_r
        state.t_r += 1
        if et == "merge_claims":
            state.m_r += 1

        # Algorithm 1, lines 10-11: pressure and deficit
        p_r     = self.params.a_c * (state.t_r ** self.params.beta_c)
        delta_r = p_r - state.m_r

        # Algorithm 1, line 12: threshold check
        if delta_r <= self.params.delta_c:
            return None

        # Threshold exceeded — trigger integration
        return self._trigger_integration(state, row, delta_r)

    def log_stats(self, run_dir: Path) -> None:
        """Write DTI run statistics to run_dir/dti_stats.json."""
        stats = {
            "params":         self.params.to_dict(),
            "n_events_seen":  self._n_events_seen,
            "n_triggers":     self._n_triggers,
            "n_cascades":     len(self._cascades),
            "trigger_log":    self._trigger_log,
            "cascade_states": [
                {
                    "root_claim_id": s.root_claim_id,
                    "t_r_final":     s.t_r,
                    "m_r_final":     s.m_r,
                    "trigger_count": s.trigger_count,
                }
                for s in self._cascades.values()
            ],
        }
        (run_dir / "dti_stats.json").write_text(json.dumps(stats, indent=2))

    # ── Internal: trigger one integration step ────────────────────────────────

    def _trigger_integration(
        self,
        state:       _CascadeState,
        trigger_row: Dict[str, Any],
        delta_r:     float,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute Algorithm 1 lines 13-18 for one DTI trigger.

        Collects B_r (branch heads), calls LLM for integration, logs the
        resulting MERGE TraceRow to the bus, restarts (t_r, M_r).
        """
        state.trigger_count += 1
        self._n_triggers    += 1

        # B_r: active branch heads (paper Eq. A4)
        branch_outputs = list(state.branch_heads.values())
        if not branch_outputs:
            return None

        # Build integration prompt (paper Appendix C)
        user_content = _build_integration_prompt(
            branch_outputs=branch_outputs,
            root_task=self._root_task or self.task_id,
        )

        # LLM call for merge integration
        t0 = time.time()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response  = self.llm.invoke([
                SystemMessage(content=_DTI_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            raw_text = response.content if hasattr(response, "content") else str(response)
            success  = True
        except Exception as e:
            raw_text = f"[DTI integration error: {e}]"
            success  = False

        latency_ms = (time.time() - t0) * 1000

        # Parse JSON output
        answer, reasoning, confidence = raw_text, "", 0.7
        try:
            text = raw_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1:
                p = json.loads(text[s:e+1])
                answer     = p.get("answer", raw_text)
                reasoning  = p.get("reasoning", "")
                confidence = float(p.get("confidence", 0.7))
                if not isinstance(answer, str):
                    answer = json.dumps(answer)
        except Exception:
            pass

        # Parent claim IDs = all branch head claim IDs (merge fan-in)
        parent_claim_ids = list(dict.fromkeys(
            b.get("claim_id") for b in branch_outputs if b.get("claim_id")
        ))

        # Merge depth = max(parent depths) + 1
        parent_depths = [b.get("claim_depth") or 0 for b in branch_outputs]
        merge_depth   = max(parent_depths) + 1

        merge_claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        message_id     = f"msg_dti_{uuid.uuid4().hex[:12]}"

        merge_row: Dict[str, Any] = {
            # ── Run metadata ────────────────────────────────────────────
            "run_id":       self.run_id,
            "benchmark":    self.benchmark,
            "task_id":      self.task_id,
            "task_family":  self.task_family,
            "topology":     self.topology,
            "seed":         self.seed,
            "num_agents":   self.num_agents,
            # ── Turn metadata ───────────────────────────────────────────
            "step_id":      trigger_row.get("step_id", 0),
            "timestamp":    time.time(),
            "agent_id":     "dti_synthesizer",
            "agent_role":   "synthesizer",
            "role":         "synthesizer",
            "message_id":   message_id,
            "neighbor_ids": [],
            # ── Claim lineage ───────────────────────────────────────────
            "claim_id":          merge_claim_id,
            "parent_claim_ids":  parent_claim_ids,
            "root_claim_id":     state.root_claim_id,
            "claim_depth":       merge_depth,
            # ── Subtask lineage (inherit from trigger row) ──────────────
            "subtask_id":           trigger_row.get("subtask_id"),
            "parent_subtask_id":    trigger_row.get("subtask_id"),
            "assigned_agent":       "dti_synthesizer",
            "subtask_status":       "active",
            "subtask_depth":        trigger_row.get("subtask_depth"),
            # ── Content ────────────────────────────────────────────────
            "reasoning_text":       reasoning,
            "final_answer_text":    None,
            "message_length_chars": len(raw_text),
            "message_length":       len(raw_text),
            "confidence":           confidence,
            # ── Event classification ───────────────────────────────────
            # Assigned directly — DTI always produces a merge event.
            # Post-hoc extractor will confirm or override.
            "event_type":       "merge_claims",
            "event_type_hint":  "merge_claims",
            "claim_status":     None,
            # ── Merge metadata ─────────────────────────────────────────
            "merge_id":                     f"merge_{uuid.uuid4().hex[:8]}",
            "merge_parent_claim_ids":       parent_claim_ids,
            "merge_num_inputs":             len(parent_claim_ids),
            "merge_num_unique_agents":      len({b.get("agent_id") for b in branch_outputs}),
            "merge_synthesizer_agent_id":   "dti_synthesizer",
            # ── Coordination signals ───────────────────────────────────
            "synthesis_of_multiple_inputs": True,
            "supports_claims":              parent_claim_ids,
            "explicit_disagreement_with":   [],
            "explicit_correction_of":       [],
            "requested_subtask_creation":   False,
            "proposed_assignee":            None,
            # ── Provenance ──────────────────────────────────────────────
            "references_used":          parent_claim_ids,
            "claims_visible":           parent_claim_ids,
            "claims_referenced":        parent_claim_ids,
            "tools_used":               [],
            "evidence_refs":            [],
            # ── Post-hoc fields (filled by dag_builder) ─────────────────
            "revision_chain_id":        None,
            "contradiction_group_id":   None,
            "secondary_root_claim_ids": [],
            # ── Performance ────────────────────────────────────────────
            "latency_ms":       round(latency_ms, 2),
            "action_success":   success,
            "tokens_input":     max(1, len(user_content) // 4),
            "tokens_output":    max(1, len(raw_text) // 4),
            # ── DTI-specific fields (for analysis and debugging) ────────
            "dti_triggered":     True,
            "dti_delta_r":       round(delta_r, 4),
            "dti_t_r_before":    state.t_r,
            "dti_m_r_before":    state.m_r,
            "dti_trigger_count": state.trigger_count,
        }

        # Log to bus — same events.jsonl, picked up by post-hoc pipeline
        self.bus.log(merge_row)

        # Algorithm 1, line 18: restart cascade segment
        # t_r ← 0, M_r ← 1  (one integration already realized)
        state.t_r = 0
        state.m_r = 1

        # Reset branch heads to the merged claim — next expansion starts
        # from the consolidated state (paper Appendix C, Eq. A5)
        state.branch_heads = {"dti_synthesizer": merge_row}

        self._trigger_log.append({
            "root_claim_id":  state.root_claim_id,
            "step_id":        trigger_row.get("step_id", 0),
            "delta_r":        round(delta_r, 4),
            "n_branches":     len(branch_outputs),
            "merge_claim_id": merge_claim_id,
            "trigger_count":  state.trigger_count,
        })

        return merge_row


# ── Parameter estimation utilities ───────────────────────────────────────────

def estimate_dti_params_from_sweep(
    sweep_data_root: Path,
    conditions:      Optional[List[str]] = None,
    beta_c:          float = 0.60,
    save_path:       Optional[Path] = None,
) -> Dict[str, DTIParams]:
    """
    Estimate DTI parameters for all condition classes from completed baseline
    sweep data.

    Run this once after the baseline sweep completes and before launching the
    DTI sweep. The resulting params dict is passed to graph_runner.run() via
    DTIParams.load().

    Args:
        sweep_data_root:  root of completed baseline sweep (e.g. data/sweep/)
        conditions:       condition keys to estimate (None = all)
        beta_c:           contradiction scaling exponent — use paper value
                          0.60 or fit from your own data
        save_path:        if provided, save to JSON for reuse

    Returns:
        {condition_key: DTIParams}
    """
    from analysis.run_pipeline import discover_run_dirs

    run_dirs_by_condition = discover_run_dirs(sweep_data_root)
    params_out: Dict[str, DTIParams] = {}

    for ckey, run_dirs in sorted(run_dirs_by_condition.items()):
        if conditions and ckey not in conditions:
            continue
        print(f"  Estimating DTI params: {ckey}  ({len(run_dirs)} runs)")
        params = DTIParams.from_baseline_traces(
            condition_key=ckey,
            run_dirs=run_dirs,
            beta_c=beta_c,
        )
        params_out[ckey] = params
        print(f"    β̂_c={params.beta_c:.3f}  a_c={params.a_c:.4f}  "
              f"δ_c={params.delta_c:.3f}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(
            {k: v.to_dict() for k, v in params_out.items()}, indent=2
        ))
        print(f"\n  Saved DTI params → {save_path}")

    return params_out


def load_dti_params(path: Path) -> Dict[str, DTIParams]:
    """Load DTI params saved by estimate_dti_params_from_sweep."""
    raw = json.loads(path.read_text())
    return {
        k: DTIParams(
            condition_key=v["condition_key"],
            beta_c=float(v["beta_c"]),
            a_c=float(v["a_c"]),
            delta_c=float(v["delta_c"]),
        )
        for k, v in raw.items()
    }