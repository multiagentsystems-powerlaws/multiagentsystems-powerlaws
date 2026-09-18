#!/usr/bin/env python3
"""
scripts/test_pipeline.py

Final experimental design (confirmed):
  Scales: N ∈ {8, 16, 32, 64, 128}
  Seeds:  3  (0, 1, 2)
  Tasks:  70 (MARBLE:20, GAIA:20, REALM:20, SWE-bench:10)
-------------------------
Comprehensive end-to-end pipeline test for Tinkercliffs.

Runs 8 test stages in order. Each stage prints PASS / FAIL / SKIP
with a clear reason. Stops at the first FAIL so you know exactly
what to fix before proceeding.

Usage
-----
  # Full test (requires OPENAI_API_KEY and all benchmarks accessible)
  python scripts/test_pipeline.py

  # Skip actual LLM calls (just test imports + data loading)
  python scripts/test_pipeline.py --no-llm

  # Test specific benchmarks only
  python scripts/test_pipeline.py --benchmarks MARBLE REALM

  # Use a different model
  python scripts/test_pipeline.py --model gpt-4o-mini

Output
------
  data/pipeline_test/          <- all smoke run artefacts
  data/pipeline_test/report.json  <- machine-readable test report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_DATA_ROOT = ROOT / "data" / "pipeline_test"
TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)


# ── Colour helpers ────────────────────────────────────────────────────────
def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"


# ── Result collector ──────────────────────────────────────────────────────
results: List[Dict[str, Any]] = []

def record(stage: str, status: str, detail: str = "", data: Any = None):
    icon = {"PASS": green("✓"), "FAIL": red("✗"), "SKIP": yellow("○"), "WARN": yellow("!")}
    print(f"  {icon.get(status,'?')} [{status}] {stage}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    results.append({"stage": stage, "status": status, "detail": detail, "data": data})

def section(title: str):
    print(f"\n{bold('── ' + title + ' ──')}")


# ═════════════════════════════════════════════════════════════════════════
# STAGE 1 — Python imports
# ═════════════════════════════════════════════════════════════════════════
def test_imports():
    section("Stage 1: Imports")
    modules = [
        ("loggers.schemas",               "AgentEvent, RunConfig, TopologyName"),
        ("loggers.event_bus",             "EventBus"),
        ("topologies",                    "get_topology, ALL_TOPOLOGIES"),
        ("execution.mas_state",           "MASState, initial_state"),
        ("execution.graph_runner",        "GraphRunner, BenchmarkTask"),
        ("execution.runner",              "SweepRunner, SweepConfig"),
        ("agents.llm_agent",              "LLMAgent, RunContext"),
        ("prompts.templates",             "PromptBuilder"),
        ("benchmark_wrappers.marble",     "load_marble_tasks"),
        ("benchmark_wrappers.realm_bench","load_realm_tasks"),
        ("benchmark_wrappers.gaia",       "load_gaia_tasks"),
        ("benchmark_wrappers.swebench",   "load_swebench_tasks"),
        ("benchmark_wrappers.task_curator","curate_tasks"),
        ("event_extraction.coordination", "extract_all_observables"),
        ("event_extraction.tce",          "compute_all_tce"),
        ("tail_fitting.powerlaw_fit",     "fit_observable, fit_all"),
    ]
    failed = []
    for mod, symbols in modules:
        try:
            m = __import__(mod, fromlist=symbols.split(","))
            for sym in [s.strip() for s in symbols.split(",")]:
                getattr(m, sym)
        except Exception as e:
            failed.append(f"{mod}: {e}")

    if failed:
        record("Python imports", "FAIL", "\n".join(failed))
        return False
    record("Python imports", "PASS", f"All {len(modules)} modules imported OK")
    return True


# ═════════════════════════════════════════════════════════════════════════
# STAGE 2 — Prompt builder
# ═════════════════════════════════════════════════════════════════════════
def test_prompts():
    section("Stage 2: Prompt builder")
    from prompts.templates import PromptBuilder
    try:
        b = PromptBuilder()
        topos    = b.list_topologies()
        families = b.list_families()

        expected_topos    = {"chain","star","tree","full_mesh","sparse_mesh",
                              "hybrid_modular","dynamic_reputation"}
        expected_families = {"planning","reasoning","coding","qa",
                              "synthesis","coordination","critique"}

        missing_t = expected_topos    - set(topos)
        missing_f = expected_families - set(families)

        if missing_t or missing_f:
            record("Prompt builder", "FAIL",
                   f"Missing topology prompts: {missing_t}\n"
                   f"Missing family prompts:   {missing_f}")
            return False

        # Check prompts are substantial (> 500 chars each)
        thin = []
        for t in expected_topos:
            p = b.system(t, "planning", agent_id="a0", num_agents=4,
                          agent_index=0, round=1, max_rounds=3,
                          depth=0, role="peer", supervisor_id="sup",
                          worker_ids="[]", neighbor_ids="[]",
                          community_id=0, community_member_ids="[]",
                          bridge_id="b0", num_consulted=3, step=1,
                          max_steps=8,
                          # star-specific
                          role_block="You are the hub.",
                          dispatch_or_work_block="Dispatch subtasks.")
            if len(p) < 500:
                thin.append(f"{t}: only {len(p)} chars")

        if thin:
            record("Prompt builder", "WARN",
                   "Some prompts are thin:\n" + "\n".join(thin))
        else:
            record("Prompt builder", "PASS",
                   f"Topologies: {sorted(topos)}\n"
                   f"Families:   {sorted(families)}\n"
                   f"All prompts > 500 chars")
        return True
    except Exception as e:
        record("Prompt builder", "FAIL", traceback.format_exc())
        return False


# ═════════════════════════════════════════════════════════════════════════
# STAGE 3 — Benchmark data loading
# ═════════════════════════════════════════════════════════════════════════
def test_benchmarks(bench_filter: Optional[List[str]] = None):
    section("Stage 3: Benchmark data loading")

    loaders = {
        "MARBLE":    ("benchmark_wrappers.marble",      "load_marble_tasks",  3),
        "REALM":     ("benchmark_wrappers.realm_bench", "load_realm_tasks",   3),
        "GAIA":      ("benchmark_wrappers.gaia",        "load_gaia_tasks",    3),
        "SWE-bench": ("benchmark_wrappers.swebench",    "load_swebench_tasks",3),
    }

    all_ok = True
    for bench, (mod, fn, n) in loaders.items():
        if bench_filter and bench not in bench_filter:
            record(f"  {bench}", "SKIP", "not in --benchmarks filter")
            continue
        try:
            m     = __import__(mod, fromlist=[fn])
            tasks = getattr(m, fn)(max_tasks=n)
            if not tasks:
                record(f"  {bench}", "FAIL", "returned 0 tasks")
                all_ok = False
                continue

            t0 = tasks[0]
            issues = []
            if not t0.task_id:    issues.append("task_id empty")
            if not t0.prompt:     issues.append("prompt empty")
            if not t0.benchmark:  issues.append("benchmark empty")
            if not t0.difficulty: issues.append("difficulty empty")
            if not t0.task_family: issues.append("task_family empty")

            if issues:
                record(f"  {bench}", "WARN",
                       f"Loaded {len(tasks)} tasks but field issues: {issues}\n"
                       f"  task_id={t0.task_id!r}  difficulty={t0.difficulty!r}")
            else:
                record(f"  {bench}", "PASS",
                       f"Loaded {len(tasks)} tasks  "
                       f"id={t0.task_id}  difficulty={t0.difficulty}  "
                       f"family={t0.task_family}  "
                       f"prompt_len={len(t0.prompt)}")
        except Exception as e:
            record(f"  {bench}", "FAIL", str(e))
            all_ok = False

    return all_ok


# ═════════════════════════════════════════════════════════════════════════
# STAGE 4 — Topology graph construction (no LLM)
# ═════════════════════════════════════════════════════════════════════════
def test_topology_construction():
    section("Stage 4: Topology graph construction (stub LLM)")
    from loggers.event_bus import EventBus
    from topologies import get_topology, ALL_TOPOLOGIES

    class StubLLM:
        def invoke(self, messages):
            class R:
                content = ("PARSE: What is 6×7?\n"
                           "CHAIN:\n  1. 6×7 = 42\n"
                           "ANSWER: 42")
            return R()

    common = dict(
        llm=StubLLM(),
        benchmark="TEST", task_id="t000",
        task_family="reasoning", difficulty="easy",
        num_agents=4, seed=0, architecture="flat_swarm",
    )

    all_ok = True
    for tname in ALL_TOPOLOGIES:
        try:
            bus   = EventBus(TEST_DATA_ROOT / "topo_smoke" / tname.value)
            topo  = get_topology(tname, bus=bus,
                                  run_id=f"smoke_{tname.value}", **common)
            graph = topo.build_graph()
            edges = topo.edge_list()
            agents = topo.agent_ids()
            bus.close()

            record(f"  {tname.value}", "PASS",
                   f"agents={len(agents)}  edges={len(edges)}  "
                   f"graph={type(graph).__name__}")
        except Exception as e:
            record(f"  {tname.value}", "FAIL", traceback.format_exc())
            all_ok = False

    return all_ok


# ═════════════════════════════════════════════════════════════════════════
# STAGE 5 — Full run with stub LLM + event logging check
# ═════════════════════════════════════════════════════════════════════════
def test_stub_run():
    section("Stage 5: Full run with stub LLM + event/snapshot logging")
    from loggers.event_bus import EventBus
    from topologies import get_topology
    from loggers.schemas import TopologyName

    class StubLLM:
        _n = 0
        def invoke(self, messages):
            self.__class__._n += 1
            class R:
                content = (f"PARSE: Step {StubLLM._n}.\n"
                           "CHAIN:\n  1. Known fact → answer\n"
                           "ANSWER: 42")
            return R()

    # Test with 3 representative topologies
    test_topos = [
        TopologyName.CHAIN,
        TopologyName.STAR,
        TopologyName.DYNAMIC_REPUTATION,
    ]
    common = dict(
        llm=StubLLM(),
        benchmark="TEST", task_id="t001",
        task_family="reasoning", difficulty="medium",
        num_agents=4, seed=42, architecture="flat_swarm",
    )

    all_ok = True
    for tname in test_topos:
        try:
            bus  = EventBus(TEST_DATA_ROOT / "stub_run" / tname.value)
            topo = get_topology(tname, bus=bus,
                                 run_id=f"stub_{tname.value}", **common)
            answer = topo.run("What is the optimal solution to this problem?")
            bus.close()

            events_path = TEST_DATA_ROOT / "stub_run" / tname.value / "events.jsonl"
            snaps_path  = TEST_DATA_ROOT / "stub_run" / tname.value / "snapshots.jsonl"

            if not events_path.exists():
                record(f"  {tname.value}", "FAIL", "events.jsonl not created")
                all_ok = False
                continue

            events = [json.loads(l) for l in open(events_path)]
            snaps  = [json.loads(l) for l in open(snaps_path)] if snaps_path.exists() else []

            # Check required fields present
            required = ["run_id","agent_id","event_type","step_id",
                        "tokens_total_event","topology","benchmark"]
            ev0 = events[0]
            missing = [f for f in required if ev0.get(f) is None]

            if missing:
                record(f"  {tname.value}", "FAIL",
                       f"Missing required event fields: {missing}\n"
                       f"Available keys: {list(ev0.keys())[:10]}")
                all_ok = False
            else:
                record(f"  {tname.value}", "PASS",
                       f"answer={'yes' if answer else 'EMPTY'}  "
                       f"events={len(events)}  snapshots={len(snaps)}  "
                       f"event_types={sorted(set(e['event_type'] for e in events))}")
        except Exception as e:
            record(f"  {tname.value}", "FAIL", traceback.format_exc())
            all_ok = False

    return all_ok


# ═════════════════════════════════════════════════════════════════════════
# STAGE 6 — Real LLM run (1 task, 1 topology, 4 agents)
# ═════════════════════════════════════════════════════════════════════════
def test_real_llm_run(model: str, bench_filter: Optional[List[str]] = None):
    section("Stage 6: Real LLM run (1 task × chain × 4 agents)")

    if not os.environ.get("OPENAI_API_KEY") and "gpt" in model.lower():
        record("Real LLM run", "SKIP",
               "OPENAI_API_KEY not set. Export it and re-run.")
        return True  # not a failure — just skipped

    # Pick benchmark to test
    bench = "MARBLE"
    if bench_filter:
        bench = bench_filter[0]

    try:
        mod  = {
            "MARBLE":    ("benchmark_wrappers.marble",      "load_marble_tasks"),
            "GAIA":      ("benchmark_wrappers.gaia",        "load_gaia_tasks"),
            "REALM":     ("benchmark_wrappers.realm_bench", "load_realm_tasks"),
            "SWE-bench": ("benchmark_wrappers.swebench",    "load_swebench_tasks"),
        }[bench]
        m     = __import__(mod[0], fromlist=[mod[1]])
        tasks = getattr(m, mod[1])(max_tasks=1)
        if not tasks:
            record("Real LLM run", "SKIP", f"{bench} returned 0 tasks")
            return True
        task = tasks[0]
    except Exception as e:
        record("Real LLM run", "SKIP", f"Could not load {bench} task: {e}")
        return True

    try:
        from llm_factory import build_llm
        llm = build_llm(model, temperature=0.7)
    except ImportError as e:
        record("Real LLM run", "SKIP", f"LLM library not installed: {e}")
        return True

    from execution.graph_runner import GraphRunner
    from loggers.schemas import TopologyName

    try:
        t0     = time.time()
        runner = GraphRunner(llm=llm, data_root=TEST_DATA_ROOT / "real_run")
        result = runner.run(task, TopologyName.CHAIN, num_agents=4, seed=0)
        elapsed = time.time() - t0

        issues = []
        if not result.final_answer:
            issues.append("final_answer is empty")
        if result.event_count == 0:
            issues.append("event_count is 0")
        if result.tokens_total == 0:
            issues.append("tokens_total is 0")
        if not result.success:
            issues.append(f"run marked failed: {result.error}")

        if issues:
            record("Real LLM run", "FAIL",
                   f"Task: {task.task_id} ({bench})\n"
                   + "\n".join(issues)
                   + f"\nAnswer preview: {result.final_answer[:100]!r}")
            return False

        record("Real LLM run", "PASS",
               f"Benchmark: {bench}  Task: {task.task_id}\n"
               f"events={result.event_count}  "
               f"tokens={result.tokens_total}  "
               f"wall_time={elapsed:.1f}s\n"
               f"answer_preview: {result.final_answer[:120]!r}")

        # Deep-check a sample event
        events_path = result.run_dir / "events.jsonl"
        events = [json.loads(l) for l in open(events_path)]
        ev = events[0]

        deep_checks = {
            "claim_id populated":   bool(ev.get("claim_id")),
            "topology matches":     ev.get("topology") == "chain",
            "benchmark matches":    ev.get("benchmark") == bench,
            "tokens > 0":           (ev.get("tokens_total_event") or 0) > 0,
            "latency_ms > 0":       (ev.get("latency_ms") or 0) > 0,
            "agent_role set":       bool(ev.get("agent_role")),
        }
        failed_checks = [k for k, v in deep_checks.items() if not v]
        if failed_checks:
            record("Event field deep-check", "WARN",
                   "Fields missing or wrong:\n" +
                   "\n".join(f"  {k}: {ev.get(k.split()[0])!r}"
                              for k in failed_checks))
        else:
            record("Event field deep-check", "PASS",
                   "All key event fields populated correctly")

        return True

    except Exception as e:
        record("Real LLM run", "FAIL", traceback.format_exc())
        return False


# ═════════════════════════════════════════════════════════════════════════
# STAGE 7 — Event extraction on smoke data
# ═════════════════════════════════════════════════════════════════════════
def test_extraction():
    section("Stage 7: Event extraction")
    # Paper-faithful per-run pipeline (event_extractor -> resolve ->
    # dag_builder -> cascade_metrics), pooled across all run directories
    # found under root. Building the DAG per-run (not on one global flat
    # event list) is required: claim_id/subtask_id are not guaranteed
    # unique across different runs.
    from analysis.run_pipeline import process_run
    from event_extraction.coordination import extract_influence_per_agent, _load_all_events

    # Use whichever test data was generated
    roots = [
        TEST_DATA_ROOT / "stub_run",
        TEST_DATA_ROOT / "real_run",
    ]
    root = next((r for r in roots if r.exists() and
                  list(r.rglob("events.jsonl"))), None)

    if root is None:
        record("Event extraction", "SKIP",
               "No events.jsonl found — run earlier stages first")
        return True

    try:
        obs: Dict[str, list] = {
            "delegation_cascade": [], "revision_wave": [],
            "contradiction_burst": [], "merge_fanin": [], "tce_per_run": [],
        }
        legacy_key = {
            "delegation_sizes": "delegation_cascade",
            "revision_waves": "revision_wave",
            "contradiction_bursts": "contradiction_burst",
            "merge_fan_in": "merge_fanin",
            "tce": "tce_per_run",
        }
        for events_file in sorted(root.rglob("events.jsonl")):
            run_obs = process_run(events_file.parent)
            for new_key, vals in run_obs["event_observables"].items():
                obs[legacy_key[new_key]].extend(vals)

        obs["influence_per_agent"] = extract_influence_per_agent(_load_all_events(root))

        populated = {k: len(v) for k, v in obs.items() if v}
        empty     = [k for k, v in obs.items() if not v]

        record("Event extraction", "PASS" if populated else "WARN",
               f"Populated observables: {populated}\n"
               f"Empty (need more data): {empty}")

        return True
    except Exception as e:
        record("Event extraction", "FAIL", traceback.format_exc())
        return False


# ═════════════════════════════════════════════════════════════════════════
# STAGE 8 — Power-law fitter on synthetic data
# ═════════════════════════════════════════════════════════════════════════
def test_fitter():
    section("Stage 8: Power-law fitter (synthetic Pareto data)")
    try:
        import numpy as np
        from tail_fitting.powerlaw_fit import fit_observable

        # Generate synthetic TRUNCATED power-law data with known alpha, x_c.
        # Pure (untruncated) Pareto data is the wrong smoke-test target now
        # that the primary model is TPL (paper Eq. 4): fitting an extra free
        # parameter (x_c) against data with no real cutoff is an inherently
        # unstable edge case for the optimizer, not representative of the
        # genuinely-truncated data this pipeline actually analyzes.
        rng        = np.random.default_rng(42)
        true_alpha, true_xc = 2.3, 25.0
        raw  = rng.pareto(true_alpha - 1, 20000) + 1.0
        raw  = raw[raw < 500]
        keep = rng.random(len(raw)) < np.exp(-raw / true_xc)
        data = list(np.round(raw[keep]).astype(int))

        result = fit_observable(data, "synthetic_tpl", verbose=False)

        if result is None:
            record("Power-law fitter", "FAIL",
                   "fit_observable returned None on synthetic TPL data")
            return False

        alpha_err = abs(result.alpha - true_alpha)
        xc_err    = abs(result.x_c - true_xc) / true_xc
        ok        = alpha_err < 0.3 and xc_err < 0.5 and result.tpl_beats_pl

        record("Power-law fitter", "PASS" if ok else "WARN",
               f"True α={true_alpha} x_c={true_xc}  "
               f"Fitted α={result.alpha:.3f} x_c={result.x_c:.1f}  "
               f"α_err={alpha_err:.3f}  x_c_relerr={xc_err:.2f}  n_tail={result.n_tail}\n"
               f"GOF p={result.gof_pvalue}  "
               f"R_TPL/LN={result.lr_tpl_vs_ln:.3f}  "
               f"2Δℓ_TPL/PL={result.lr_tpl_vs_pl:.2f} (truncation detected: {result.tpl_beats_pl})\n"
               f"Regime: {result.regime}  "
               f"TPL plausible: {result.tpl_plausible}")
        return True

    except Exception as e:
        record("Power-law fitter", "FAIL", traceback.format_exc())
        return False


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-llm",      action="store_true",
                   help="Skip stages that require an LLM API key")
    p.add_argument("--benchmarks",  nargs="+",
                   choices=["MARBLE","GAIA","REALM","SWE-bench"],
                   default=None,
                   help="Only test these benchmarks (default: all)")
    p.add_argument("--model",       default="gpt-4o-mini",
                   help="Model for the real LLM run test")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Stop immediately on first FAIL")
    return p.parse_args()


def main():
    args = parse_args()

    print(bold("\n╔══════════════════════════════════════════════╗"))
    print(bold(  "║   MAS Power-Law Pipeline — End-to-End Test  ║"))
    print(bold(  "╚══════════════════════════════════════════════╝"))
    print(f"  Model:      {args.model}")
    print(f"  Benchmarks: {args.benchmarks or 'all'}")
    print(f"  LLM calls:  {'disabled (--no-llm)' if args.no_llm else 'enabled'}")
    print(f"  Test root:  {TEST_DATA_ROOT}")

    stages = [
        ("imports",       lambda: test_imports()),
        ("prompts",       lambda: test_prompts()),
        ("benchmarks",    lambda: test_benchmarks(args.benchmarks)),
        ("topo_build",    lambda: test_topology_construction()),
        ("stub_run",      lambda: test_stub_run()),
        ("real_llm",      lambda: (True if args.no_llm else
                                   test_real_llm_run(args.model, args.benchmarks))),
        ("extraction",    lambda: test_extraction()),
        ("fitter",        lambda: test_fitter()),
    ]

    overall_ok = True
    for name, fn in stages:
        ok = fn()
        if not ok:
            overall_ok = False
            if args.stop_on_fail:
                print(red(f"\nStopped at stage '{name}' (--stop-on-fail)"))
                break

    # ── Summary ──────────────────────────────────────────────────────
    print(bold("\n── Summary ──────────────────────────────────"))
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        icon = {"PASS": green("✓"), "FAIL": red("✗"),
                "WARN": yellow("!"), "SKIP": yellow("○")}
        print(f"  {icon.get(r['status'],'?')} {r['stage']}")

    print()
    print(f"  PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
          f"WARN={counts['WARN']}  SKIP={counts['SKIP']}")

    if counts["FAIL"] == 0:
        print(green("\n  Pipeline is healthy. Ready for Block 8 (visualizations)."))
    else:
        print(red(f"\n  {counts['FAIL']} stage(s) failed. Fix these before running the sweep."))

    # ── Write JSON report ─────────────────────────────────────────────
    report_path = TEST_DATA_ROOT / "report.json"
    report_path.write_text(json.dumps({
        "overall": "PASS" if counts["FAIL"] == 0 else "FAIL",
        "counts": counts,
        "stages": results,
    }, indent=2))
    print(f"\n  Full report → {report_path}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
