"""
execution/graph_runner.py
--------------------------
Orchestrates a single experimental run.

Scoring per benchmark:
  GAIA      — normalized exact-match against gold (validation split has labels)
  SWE-bench — patch saved to predictions JSONL; Docker harness runs post-sweep
  MARBLE    — no gold answers; event-derived metrics only
  REALM     — no gold answers; event-derived metrics only

H2 metric suite (_analyze_events):
  Completeness:
    completion_ratio          = completed_subtasks / total_subtasks
    coherence_score           = 1 - (contradicted / total_claims)
    integration_score         = merged_terminal_claims / terminal_claims
    claim_participation_rate  = claims_in_merge_chains / total_claims
    resolution_rate           = claims_resolved_or_merged / total_claims

  Normalized coordination intensity (topology-comparable):
    revisions_per_claim
    merges_per_claim
    contradictions_per_claim
    endorsements_per_claim

  Efficiency:
    success_per_token
    completion_per_token
    quality_adjusted_efficiency
    tokens_per_event
    events_per_agent

PATCH NOTES (aligned with new pipeline):
  - Added: from tools.tools import get_tool_names_for_benchmark
  - Added: topo.set_tool_names(tool_names) after get_topology()
  - _analyze_events: tokens_total uses message_length (TraceRow field)
  - _analyze_events: merge parent IDs read from parent_claim_ids (TraceRow)
  - _analyze_events: final claim detection uses role + final_answer_text
  All other logic preserved exactly.
"""

from __future__ import annotations

import json
import re
import string
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loggers.event_bus import EventBus
from loggers.schemas import (
    RunConfig, RunMetadata, TopologyName,
    RoutingStrategy, MemoryType,
)
from topologies import get_topology

# PATCH: benchmark-native tool registry
from tools.tools import get_tool_names_for_benchmark


# ────────────────────────────────────────────────────────────────
# Architecture label map — one per topology
# ────────────────────────────────────────────────────────────────

_TOPOLOGY_ARCHITECTURE: Dict[str, str] = {
    "chain":              "sequential_pipeline",
    "star":               "hub_and_spoke",
    "tree":               "hierarchical_tree",
    "full_mesh":          "scalable_full_mesh",
    "sparse_mesh":        "sparse_mesh",
    "hybrid_modular":     "modular_bridge_integrator",
    "dynamic_reputation": "dynamic_reputation",
}


# ────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    task_id:      str
    benchmark:    str
    task_family:  str
    difficulty:   str
    prompt:       str
    gold_answer:  Optional[str]  = None
    metadata:     Dict[str, Any] = field(default_factory=dict)
    requires_tools:     bool = False
    requires_synthesis: bool = False


@dataclass
class RunResult:
    run_id:       str
    task_id:      str
    benchmark:    str
    topology:     str
    num_agents:   int
    seed:         int
    success:      bool
    score:        Optional[float]
    final_answer: str
    tokens_total: int
    wall_time_s:  float
    event_count:  int
    run_dir:      Path
    error:        Optional[str] = None


# ────────────────────────────────────────────────────────────────
# Event-log analysis — all H2 metrics, single pass, no LLM
# ────────────────────────────────────────────────────────────────

def _analyze_events(events_path: Path) -> Dict[str, Any]:
    """
    Single-pass scan of events.jsonl.
    Reads TraceRow dicts (new pipeline). Field mapping vs old AgentEvent:
      tokens_total_event  → message_length  (char count proxy)
      merge_parent_claim_ids → parent_claim_ids (when event_type==merge_claims)
      claim_type=="final_claim" → role in (synthesizer, hub) + final_answer_text
    Returns a dict of all H2 metrics — deterministic, no LLM calls.
    """
    if not events_path.exists():
        return {}

    tokens_total     = 0
    event_count      = 0
    n_revisions      = 0
    n_contradictions = 0
    n_merges         = 0
    n_endorsements   = 0
    n_proposals      = 0
    n_finalizations  = 0
    n_delegations    = 0
    n_completions    = 0

    subtask_ids_created    = set()
    subtask_ids_completed  = set()

    claim_ids_seen         = set()
    claim_ids_merged       = set()
    claim_ids_contradicted = set()
    claim_ids_final        = set()
    merge_parent_ids       = set()

    for line in open(events_path):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue

        event_count  += 1

        # PATCH: TraceRow uses message_length (char count), not tokens_total_event
        tokens_total += ev.get("message_length") or 0

        ev_type = (ev.get("event_type") or "").lower()

        if ev_type == "revise_claim":
            n_revisions += 1
        elif ev_type == "contradict_claim":
            n_contradictions += 1
        elif ev_type == "merge_claims":
            n_merges += 1
            # PATCH: TraceRow uses parent_claim_ids (not merge_parent_claim_ids)
            for pid in (ev.get("parent_claim_ids") or []):
                merge_parent_ids.add(pid)
            if ev.get("claim_id"):
                claim_ids_merged.add(ev["claim_id"])
        elif ev_type == "endorse_claim":
            n_endorsements += 1
        elif ev_type == "propose_claim":
            n_proposals += 1
        elif ev_type == "finalize_answer":
            n_finalizations += 1
        elif ev_type == "delegate_subtask":
            n_delegations += 1
        elif ev_type == "complete_subtask":
            n_completions += 1

        cid = ev.get("claim_id")
        if cid and ev_type != "endorse_claim":
            claim_ids_seen.add(cid)

            # PATCH: TraceRow has no claim_type field.
            # Final claims = synthesizer/hub rows that have a final_answer_text.
            if (ev.get("role") in ("synthesizer", "hub")
                    and ev.get("final_answer_text")):
                claim_ids_final.add(cid)

            if ev.get("claim_status") == "contradicted":
                claim_ids_contradicted.add(cid)

        sid = ev.get("subtask_id")
        if sid:
            subtask_ids_created.add(sid)
        if ev.get("subtask_status") == "complete" and sid:
            subtask_ids_completed.add(sid)

    # ── Completeness ──────────────────────────────────────────────
    n_sub_total      = len(subtask_ids_created)
    n_sub_completed  = len(subtask_ids_completed)
    completion_ratio = (n_sub_completed / n_sub_total) if n_sub_total > 0 else 0.0

    # ── Coherence ─────────────────────────────────────────────────
    n_claims_total    = len(claim_ids_seen)
    n_unresolved      = len(claim_ids_contradicted)
    coherence_score   = (1.0 - n_unresolved / n_claims_total) if n_claims_total > 0 else 1.0

    # ── Integration score ─────────────────────────────────────────
    terminal_claims   = claim_ids_final - merge_parent_ids
    n_terminal        = len(terminal_claims)
    merged_terminals  = terminal_claims & claim_ids_merged
    integration_score = (len(merged_terminals) / n_terminal) if n_terminal > 0 else 0.0

    # ── Claim participation rate ──────────────────────────────────
    claims_in_merge_chains   = merge_parent_ids | claim_ids_merged
    claim_participation_rate = (
        len(claims_in_merge_chains) / n_claims_total if n_claims_total > 0 else 0.0
    )

    # ── Resolution rate ───────────────────────────────────────────
    resolved_claims = merge_parent_ids | claim_ids_merged | claim_ids_final
    resolution_rate = (
        len(resolved_claims) / n_claims_total if n_claims_total > 0 else 0.0
    )

    # ── Normalized coordination intensity ─────────────────────────
    def _rate(n):
        return round(n / n_claims_total, 4) if n_claims_total > 0 else 0.0

    return {
        "tokens_total":               tokens_total,
        "messages_total":             event_count,
        "num_revisions_total":        n_revisions,
        "num_contradictions_total":   n_contradictions,
        "num_merges_total":           n_merges,
        "num_endorsements_total":     n_endorsements,
        "num_subtasks_total":         n_sub_total,
        "num_subtasks_completed":     n_sub_completed,
        "num_subtasks_open_final":    max(0, n_sub_total - n_sub_completed),
        "completion_ratio":           round(completion_ratio, 4),
        "num_claims_total":           n_claims_total,
        "num_claims_merged":          len(claim_ids_merged),
        "num_claims_unresolved_final": n_unresolved,
        "coherence_score":            round(coherence_score, 4),
        "integration_score":          round(integration_score, 4),
        "claim_participation_rate":   round(claim_participation_rate, 4),
        "resolution_rate":            round(resolution_rate, 4),
        "revisions_per_claim":        _rate(n_revisions),
        "merges_per_claim":           _rate(n_merges),
        "contradictions_per_claim":   _rate(n_contradictions),
        "endorsements_per_claim":     _rate(n_endorsements),
        "num_coordination_events_total": event_count,
    }


# ────────────────────────────────────────────────────────────────
# GAIA scoring — normalized exact match (official metric)
# ────────────────────────────────────────────────────────────────

def _normalize_gaia(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    s = re.sub(r'\s+', ' ', s).strip()
    def _norm_num(m):
        try:
            n = float(m.group())
            return str(int(n)) if n == int(n) else str(n)
        except Exception:
            return m.group()
    s = re.sub(r'\d+\.?\d*', _norm_num, s)
    return s


def _score_gaia(answer: str, task: BenchmarkTask) -> Tuple[Optional[float], Dict]:
    gold = task.gold_answer
    if not gold:
        return None, {"gaia_exact_match": None, "gaia_rubric_score": None}
    match = (_normalize_gaia(answer) == _normalize_gaia(gold))
    score = 1.0 if match else 0.0
    return score, {"gaia_exact_match": match, "gaia_rubric_score": score}


# ────────────────────────────────────────────────────────────────
# SWE-bench — save patch for offline Docker harness
# ────────────────────────────────────────────────────────────────

def _score_swebench(answer: str, task: BenchmarkTask, run_dir: Path, run_id: str) -> Tuple[Optional[float], Dict]:
    diff_match = re.search(r'(---\s+\S+.*?(?=\Z|\n(?![\+\-@ ])|\Z))', answer, re.DOTALL)
    patch = diff_match.group(1).strip() if diff_match else answer.strip()

    prediction = {
        "instance_id":        task.task_id,
        "model_patch":        patch,
        "model_name_or_path": run_id,
    }
    (run_dir / "swe_prediction.json").write_text(json.dumps(prediction, indent=2))
    sweep_preds = run_dir.parents[3] / "swe_predictions.jsonl"
    with open(sweep_preds, "a") as f:
        f.write(json.dumps(prediction) + "\n")

    ftp = task.metadata.get("fail_to_pass", [])
    return None, {
        "swe_patch_applied":  bool(patch),
        "swe_tests_passed":   None,
        "swe_tests_total":    len(ftp) if isinstance(ftp, list) else None,
        "swe_files_modified": len(set(re.findall(r'(?:---|\+\+\+)\s+(\S+)', patch))),
    }


def _score_no_gold(answer: str, task: BenchmarkTask) -> Tuple[Optional[float], Dict]:
    return None, {}


def _score_task(answer: str, task: BenchmarkTask, run_dir: Path, run_id: str) -> Tuple[Optional[float], Dict]:
    b = task.benchmark.upper()
    if "GAIA" in b: return _score_gaia(answer, task)
    if "SWE"  in b: return _score_swebench(answer, task, run_dir, run_id)
    return _score_no_gold(answer, task)


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

class GraphRunner:

    def __init__(
        self,
        *,
        llm,
        data_root:        Path  = Path("data/runs"),
        architecture:     str   = "",
        routing_strategy: RoutingStrategy = RoutingStrategy.PLANNER_ASSIGNED,
        memory_type:      MemoryType      = MemoryType.SLIDING_WINDOW,
        snapshot_every:   int   = 5,
        max_steps:        int   = 50,
        model_name:       str   = "gpt-4o-mini",
        temperature:      float = 0.7,
    ) -> None:
        self.llm              = llm
        self.data_root        = Path(data_root)
        self.architecture     = architecture
        self.routing_strategy = routing_strategy
        self.memory_type      = memory_type
        self.snapshot_every   = snapshot_every
        self.max_steps        = max_steps
        self.model_name       = model_name
        self.temperature      = temperature

    def run(
        self,
        task:       BenchmarkTask,
        topology:   TopologyName | str,
        num_agents: int,
        seed:       int,
        run_id:     Optional[str] = None,
    ) -> RunResult:
        tname  = TopologyName(topology) if isinstance(topology, str) else topology
        run_id = run_id or (
            f"{task.benchmark}__{tname.value}__n{num_agents}__s{seed}__{task.task_id}"
        )

        architecture = self.architecture or _TOPOLOGY_ARCHITECTURE.get(
            tname.value, tname.value
        )

        run_dir = (
            self.data_root / tname.value
            / f"n{num_agents}" / f"s{seed}" / task.task_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        bus = EventBus(run_dir)
        config = RunConfig(
            run_id=run_id,
            benchmark=task.benchmark,
            task_id=task.task_id,
            task_family=task.task_family,
            difficulty=task.difficulty,
            task_requires_tools=task.requires_tools,
            task_requires_synthesis=task.requires_synthesis,
            topology=tname,
            architecture=architecture,
            routing_strategy=self.routing_strategy,
            memory_type=self.memory_type,
            num_agents=num_agents,
            max_steps=self.max_steps,
            seed=seed,
            run_seed=seed,
            task_seed=seed,
            topology_seed=seed,
            model_name=self.model_name,
            temperature=self.temperature,
        )
        bus.write_run_config(config)

        t0                 = time.time()
        error              = None
        answer             = ""
        score              = None
        bench_extras: Dict[str, Any] = {}
        activation_summary = {}

        try:
            topo = get_topology(
                tname,
                llm=self.llm,
                bus=bus,
                run_id=run_id,
                benchmark=task.benchmark,
                task_id=task.task_id,
                task_family=task.task_family,
                difficulty=task.difficulty,
                num_agents=num_agents,
                seed=seed,
                architecture=architecture,
                snapshot_every=self.snapshot_every,
            )

            # PATCH: inject benchmark-native tool names into topology.
            # Same tool set for all agents, all roles, all topologies, all N.
            # Topology passes these to AgentContextSpec.available_tools.
            tool_names = get_tool_names_for_benchmark(task.benchmark)
            if hasattr(topo, "set_tool_names"):
                topo.set_tool_names(tool_names)

            answer             = topo.run(task.prompt)
            activation_summary = getattr(topo, "_activation_summary", {})
            score, bench_extras = _score_task(answer, task, run_dir, run_id)

        except Exception:
            error = traceback.format_exc()

        wall_time    = time.time() - t0
        ev_metrics   = _analyze_events(run_dir / "events.jsonl")
        tokens_total = ev_metrics.get("tokens_total", 0)
        event_count  = ev_metrics.get("messages_total", 0)

        def _safe_div(n, d):
            return round(n / d, 6) if d and d > 0 and n is not None else None

        tokens_per_event = _safe_div(tokens_total, event_count)
        events_per_agent = _safe_div(event_count, num_agents)

        meta = RunMetadata(
            run_id=run_id,
            task_success=(score > 0) if score is not None else None,
            task_score=score,
            swe_patch_applied=bench_extras.get("swe_patch_applied"),
            swe_tests_passed=bench_extras.get("swe_tests_passed"),
            swe_tests_total=bench_extras.get("swe_tests_total"),
            swe_files_modified=bench_extras.get("swe_files_modified"),
            gaia_exact_match=bench_extras.get("gaia_exact_match"),
            gaia_rubric_score=bench_extras.get("gaia_rubric_score"),
            gaia_tools_used=bench_extras.get("gaia_tools_used"),
            marble_subgoals_completed=bench_extras.get("marble_subgoals_completed"),
            marble_constraints_satisfied=bench_extras.get("marble_constraints_satisfied"),
            marble_team_objective_met=bench_extras.get("marble_team_objective_met"),
            realm_plan_valid=bench_extras.get("realm_plan_valid"),
            realm_recovered_from_disruption=bench_extras.get("realm_recovered_from_disruption"),
            realm_num_replans=bench_extras.get("realm_num_replans"),
            realm_dependency_satisfaction_rate=bench_extras.get("realm_dependency_satisfaction_rate"),
            num_subtasks_total=ev_metrics.get("num_subtasks_total", 0),
            num_subtasks_completed=ev_metrics.get("num_subtasks_completed", 0),
            num_subtasks_open_final=ev_metrics.get("num_subtasks_open_final", 0),
            completion_ratio=ev_metrics.get("completion_ratio", 0.0),
            num_claims_total=ev_metrics.get("num_claims_total", 0),
            num_claims_merged=ev_metrics.get("num_claims_merged", 0),
            num_claims_unresolved_final=ev_metrics.get("num_claims_unresolved_final", 0),
            coherence_score=ev_metrics.get("coherence_score", 1.0),
            num_revisions_total=ev_metrics.get("num_revisions_total", 0),
            num_contradictions_total=ev_metrics.get("num_contradictions_total", 0),
            num_merges_total=ev_metrics.get("num_merges_total", 0),
            num_endorsements_total=ev_metrics.get("num_endorsements_total", 0),
            integration_score=ev_metrics.get("integration_score", 0.0),
            tokens_total=tokens_total,
            messages_total=event_count,
            num_coordination_events_total=event_count,
            wall_time_seconds=round(wall_time, 2),
            success_per_token=_safe_div(score, tokens_total),
            completion_per_token=_safe_div(ev_metrics.get("completion_ratio"), tokens_total),
            quality_adjusted_efficiency=_safe_div(score, wall_time),
            num_unique_agents_activated=activation_summary.get("unique_agents_touched", num_agents),
            unique_agents_touched=activation_summary.get("unique_agents_touched", num_agents),
            mean_active_per_step=activation_summary.get("mean_active_per_step", float(num_agents)),
            activation_rate=activation_summary.get("activation_rate", 1.0),
            active_agents_per_step=activation_summary.get("active_agents_per_step", {}),
            extra={
                **({"error": error} if error else {}),
                "claim_participation_rate":   ev_metrics.get("claim_participation_rate", 0.0),
                "resolution_rate":            ev_metrics.get("resolution_rate", 0.0),
                "revisions_per_claim":        ev_metrics.get("revisions_per_claim", 0.0),
                "merges_per_claim":           ev_metrics.get("merges_per_claim", 0.0),
                "contradictions_per_claim":   ev_metrics.get("contradictions_per_claim", 0.0),
                "endorsements_per_claim":     ev_metrics.get("endorsements_per_claim", 0.0),
                "tokens_per_event":           tokens_per_event,
                "events_per_agent":           events_per_agent,
                "architecture":               architecture,
            },
        )

        bus.flush_run_outcome(meta)
        bus.close()

        return RunResult(
            run_id=run_id,
            task_id=task.task_id,
            benchmark=task.benchmark,
            topology=tname.value,
            num_agents=num_agents,
            seed=seed,
            success=(error is None),
            score=score,
            final_answer=answer,
            tokens_total=tokens_total,
            wall_time_s=round(wall_time, 2),
            event_count=event_count,
            run_dir=run_dir,
            error=error,
        )