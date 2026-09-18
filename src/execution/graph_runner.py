"""
execution/graph_runner.py
--------------------------
Orchestrates a single experimental run.

Pipeline per run:
  1. TaskExpander.build() → TaskTree (K seeds, K*M expanded nodes, root)
  2. get_topology() → topology graph
  3. For each node in execution_pool + [root_node]:
       topo.run(node.description + dep_context) → node_answer
       All TraceRows written to shared events.jsonl with node.node_id as subtask_id
  4. Root node answer = final_answer
  5. _analyze_events() → H2 metrics from events.jsonl
  6. RunMetadata flushed to run_metadata.json

Task tree integration (paper Section H):
  - TaskExpander stamps node.node_id as subtask_id in every TraceRow
  - dependency_dag governs prior_outputs visible per node (dep_context)
  - seed_nodes carry ground truth; accuracy scored against them
  - For smoke tests: use_llm=False (synthetic expansion)
  - For full sweep: use_llm=True, pass real benchmark pool

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
    claim_ids_with_children = set()

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
                claim_ids_with_children.add(pid)
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

            # Treat any claim attached to the root task or any claim-bearing row with
            # final_answer_text as a candidate final/root claim.
            if ev.get("task_id") == "root_000" or ev.get("final_answer_text"):
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
    # Terminal claims = claims that are never used as parents of later claims.
    terminal_claims   = claim_ids_seen - claim_ids_with_children
    # Prefer true DAG terminals; if none exist, fall back to root/final claims.
    if not terminal_claims:
        terminal_claims = set(claim_ids_final)
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

            # ── Task tree expansion (paper Section H) ──────────────────────
            # TaskExpander.build() produces:
            #   execution_pool: K*M nodes agents actually work on
            #   root_node:      synthesis node (depends on all seeds)
            #   dependency_dag: sparse DAG edges between nodes
            #   agent_allocation: {node_id: agent_budget}
            #
            # Each node becomes one topo.run() call. The topology runs its
            # full N-agent graph over each node's description, with the node's
            # node_id stamped as subtask_id in every TraceRow via the bus.
            # All runs share the same bus → single events.jsonl with full
            # subtask lineage for dag_builder to reconstruct.
            #
            # The bus stamps subtask_id per-call via topo._initial_state(),
            # which reads it from the topology's current subtask context.
            # We inject it by setting topo.current_subtask_id before each call.

            from benchmark_wrappers.task_expander import (
                TaskExpander, build_node_prompt
            )

            expander = TaskExpander(
                benchmark=task.benchmark,
                domain=task.task_family,
                seed=seed,
                model="gpt-4o-mini",
            )

            # Build tree using the task as a single-item benchmark pool.
            # For smoke tests / single-task runs this is correct.
            # The full sweep passes the real benchmark pool externally.
            task_pool = [{
                "node_id":     task.task_id,
                "description": task.prompt,
                "ground_truth": task.gold_answer,
            }]
            task_tree = expander.build(
                N=num_agents,
                benchmark_pool=task_pool,
                use_llm=False,   # synthetic expansion for single-task smoke runs;
                                 # full sweep uses use_llm=True with real pool
            )

            # Write task tree to run dir for inspection
            (run_dir / "task_tree.json").write_text(
                json.dumps(task_tree.to_dict(), indent=2)
            )

            # ── Execute agents over task tree nodes ─────────────────────────
            # Seeds are NOT executed — they are ground truth evaluation anchors.
            # execution_pool contains expanded_nodes only.
            # Execution order respects DAG dependencies (topological sort).
            # Root runs last to synthesise across all expanded outputs.
            #
            # subtask_id injection:
            #   topo.task_id = node.node_id before each topo.run() call so that
            #   every TraceRow logged by _acall_llm carries the correct subtask_id.
            #
            # Prompt construction:
            #   build_node_prompt() uses the structured TaskNode fields
            #   (objective, source_facts, required_constraints) to assemble a
            #   grounded prompt — not free-text description concatenation.

            node_answers: Dict[str, str] = {}
            answer = ""

            def _topo_sort_pool(nodes):
                """Sort execution_pool nodes so dependencies run before dependents."""
                from collections import deque
                pool_ids  = {n.node_id for n in nodes}
                in_deg    = {n.node_id: 0 for n in nodes}
                children  = {n.node_id: [] for n in nodes}
                for n in nodes:
                    for dep_id in n.depends_on:
                        if dep_id in pool_ids:
                            in_deg[n.node_id] += 1
                            children[dep_id].append(n.node_id)
                queue   = deque(nid for nid, d in in_deg.items() if d == 0)
                ordered = []
                nmap    = {n.node_id: n for n in nodes}
                while queue:
                    nid = queue.popleft()
                    ordered.append(nmap[nid])
                    for cid in children[nid]:
                        in_deg[cid] -= 1
                        if in_deg[cid] == 0:
                            queue.append(cid)
                # Append any remaining (handles cycles gracefully)
                seen = {n.node_id for n in ordered}
                ordered += [n for n in nodes if n.node_id not in seen]
                return ordered

            ordered_pool   = _topo_sort_pool(task_tree.execution_pool)
            execution_order = ordered_pool + [task_tree.root_node]

            for node in execution_order:
                # Inject node identity so every TraceRow gets task_id = node.node_id
                topo.task_id     = node.node_id
                topo.task_family = task.task_family

                # Collect outputs from dependency nodes already completed
                dep_outputs: Dict[str, str] = {
                    dep_id: node_answers[dep_id]
                    for dep_id in node.depends_on
                    if dep_id in node_answers
                }

                # Build structured, grounded prompt from TaskNode fields
                node_prompt = build_node_prompt(
                    node, dep_outputs=dep_outputs or None
                )

                node_answer = topo.run(node_prompt)
                node_answers[node.node_id] = node_answer

                if node.node_type == "root":
                    answer = node_answer

                # ── Emit completion event ────────────────────────────────────
                # This is required for _analyze_events to compute:
                #   subtask_ids_completed, completion_ratio, n_completions.
                # Without it every run shows 0 completed subtasks regardless
                # of whether execution succeeded.
                # A node is "complete" if it produced a non-empty answer.
                _subtask_status = (
                    "complete"
                    if node_answer and node_answer.strip()
                    else "failed"
                )
                bus.log({
                    # Run identity
                    "run_id":       run_id,
                    "benchmark":    task.benchmark,
                    "task_id":      task.task_id,
                    "task_family":  task.task_family,
                    "topology":     tname.value,
                    "seed":         seed,
                    "num_agents":   num_agents,
                    # Event classification
                    "event_type":       "complete_subtask",
                    "subtask_status":   _subtask_status,
                    # Subtask identity
                    "subtask_id":       node.node_id,
                    "parent_subtask_id": node.seed_parent_id,
                    "agent_id":         "runner",
                    "agent_role":       "runner",
                    "role":             "runner",
                    "step_id":          len(node_answers),  # completion order
                    "timestamp":        time.time(),
                    "message_id":       f"completion_{node.node_id}",
                    "neighbor_ids":     [],
                    # Claim fields (null — completion events are not claim events)
                    "claim_id":          None,
                    "parent_claim_ids":  [],
                    "root_claim_id":     None,
                    "claim_depth":       None,
                    # Content
                    "reasoning_text":       "",
                    "final_answer_text":    node_answer if node.node_type == "root" else None,
                    "message_length":       len(node_answer) if node_answer else 0,
                    "message_length_chars": len(node_answer) if node_answer else 0,
                    "confidence":           1.0 if _subtask_status == "complete" else 0.0,
                    # Coordination signals (none for completion events)
                    "synthesis_of_multiple_inputs": False,
                    "supports_claims":              [],
                    "explicit_disagreement_with":   [],
                    "explicit_correction_of":       [],
                    "requested_subtask_creation":   False,
                    "proposed_assignee":            None,
                    # Performance
                    "tokens_input":   0,
                    "tokens_output":  len(node_answer) // 4 if node_answer else 0,
                    "latency_ms":     0.0,
                    "action_success": _subtask_status == "complete",
                    # Post-hoc fields
                    "revision_chain_id":        None,
                    "contradiction_group_id":   None,
                    "secondary_root_claim_ids": [],
                    "subtask_depth":            None,
                    "references_used":          [],
                    "claims_visible":            [],
                    "claims_referenced":         [],
                    "tools_used":               [],
                    "evidence_refs":            [],
                })

            # Final answer is root synthesis; fall back to last expanded node
            if not answer and ordered_pool:
                answer = node_answers.get(ordered_pool[-1].node_id, "")

            # ── Evaluation bridge (fix for task_expander evaluate_accuracy) ──
            # Seeds are not executed, so seed.agent_answer must be set here
            # before _score_task / evaluate_accuracy can run.
            # Strategy: for single-seed runs (K=1) assign the root answer directly.
            # For K>1: assign the root answer to all seeds (it synthesises all of
            # them). Per-seed decomposition would require answer attribution logic
            # that is out of scope here — root-level accuracy is the correct metric
            # for multi-seed runs anyway.
            for seed_node in task_tree.seed_nodes:
                seed_node.agent_answer = answer

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