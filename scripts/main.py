#!/usr/bin/env python3
"""
main.py — MAS Power-Laws Full Pipeline
========================================
Single entry point for the complete experiment.

Stages (all run in sequence unless --stages overrides):
  1. sweep      Run LLM multi-agent sweep
  2. extract    Extract H1/H2/H3 metrics from event logs
  3. analyse    Fit power laws + alternatives
  4. tables     Generate LaTeX + CSV paper tables
  5. figures    Generate all paper figures

Usage — full sweep (one GPU per benchmark):
  python main.py --mode full --benchmark MARBLE \\
      --data-root /scratch/kavanav/mas-results/MARBLE

  python main.py --mode full --benchmark GAIA \\
      --data-root /scratch/kavanav/mas-results/GAIA

  ... (REALM, SWE-bench same pattern)

Usage — post-sweep analysis (after all GPUs finish):
  python main.py --mode analyse \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA \\
                   /scratch/.../REALM  /scratch/.../SWE-bench \\
      --out /scratch/kavanav/mas-results/combined

Usage — smoke test (all topologies, 2 seeds, 8 agents, 2 tasks/bench):
  python main.py --mode smoke \\
      --data-root data/smoke

Options:
  --mode          full | analyse | smoke  (default: full)
  --benchmark     MARBLE | GAIA | REALM | SWE-bench  (for sweep)
  --data-root     Per-benchmark sweep output root
  --data-roots    Multiple roots for post-sweep analysis
  --out           Combined analysis output dir
  --scales        Agent counts (default: 8 16 32 64 128)
  --seeds         Seeds (default: 0 1 2)
  --max-tasks     Tasks per benchmark (default: 10)
  --max-concurrent Parallel runs (default: 4)
  --model         LLM model name (default: gpt-4o-mini)
  --stages        Subset of stages to run (default: all)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

# Support running from repo root (python main.py) or scripts/ (python scripts/main.py)
ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

ALL_BENCHMARKS = ["MARBLE", "GAIA", "REALM", "SWE-bench"]
ALL_STAGES     = ["sweep", "extract", "analyse", "tables", "figures"]

SMOKE_SCALES   = [8]
SMOKE_SEEDS    = [0, 1]
SMOKE_TASKS    = 2          # per benchmark
SMOKE_STEPS    = 20
SMOKE_CONCURRENT = 2

FULL_SCALES    = [8, 16, 32, 64, 128, 256, 512]
FULL_SEEDS     = [0, 1, 2, 3, 4]
FULL_TASKS     = 5
FULL_STEPS     = 20
FULL_CONCURRENT = 4


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAS Power-Laws end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", default="full",
                   choices=["full", "analyse", "smoke"],
                   help="Pipeline mode (default: full)")
    p.add_argument("--benchmark", "-b", nargs="+",
                   choices=ALL_BENCHMARKS,
                   help="Benchmark(s) for sweep (full/smoke modes)")
    p.add_argument("--all-benchmarks", action="store_true",
                   help="Run all 4 benchmarks in sequence (single GPU)")
    p.add_argument("--data-root", default=None,
                   help="Per-benchmark sweep output root")
    p.add_argument("--data-roots", nargs="+", default=None,
                   help="Multiple roots for post-sweep analysis")
    p.add_argument("--out", default=None,
                   help="Combined analysis output dir (default: <data-root>/combined)")
    p.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None,
                   help="Subset of stages to run (default: all for mode)")
    p.add_argument("--scales", type=int, nargs="+", default=None)
    p.add_argument("--seeds",  type=int, nargs="+", default=None)
    p.add_argument("--max-tasks",     type=int, default=None, dest="max_tasks")
    p.add_argument("--max-steps",     type=int, default=None, dest="max_steps")
    p.add_argument("--max-concurrent",type=int, default=None, dest="max_concurrent")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--resume", action="store_true",
                   help="Skip runs where run_metadata.json already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without running")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Config resolution
# ─────────────────────────────────────────────────────────────────

def resolve_config(args) -> dict:
    """Merge mode defaults with explicit args."""
    mode = args.mode

    if mode == "smoke":
        scales     = args.scales      or SMOKE_SCALES
        seeds      = args.seeds       or SMOKE_SEEDS
        max_tasks  = args.max_tasks   or SMOKE_TASKS
        max_steps  = args.max_steps   or SMOKE_STEPS
        concurrent = args.max_concurrent or SMOKE_CONCURRENT
        benchmarks = args.benchmark or ALL_BENCHMARKS
        data_root  = Path(args.data_root or "data/smoke")
    else:
        scales     = args.scales      or FULL_SCALES
        seeds      = args.seeds       or FULL_SEEDS
        max_tasks  = args.max_tasks   or FULL_TASKS
        max_steps  = args.max_steps   or FULL_STEPS
        concurrent = args.max_concurrent or FULL_CONCURRENT
        benchmarks = args.benchmark or (ALL_BENCHMARKS if args.all_benchmarks else None)
        data_root  = Path(args.data_root or "data/runs")

    # data-roots for analysis
    if args.data_roots:
        data_roots = [Path(r) for r in args.data_roots]
    else:
        data_roots = [data_root]

    out = Path(args.out) if args.out else data_root / "combined"

    # Stages
    if args.stages:
        stages = args.stages
    elif mode == "analyse":
        stages = ["extract", "analyse", "tables", "figures"]
    else:
        stages = ALL_STAGES

    return {
        "mode":       mode,
        "benchmarks": benchmarks,
        "scales":     scales,
        "seeds":      seeds,
        "max_tasks":  max_tasks,
        "max_steps":  max_steps,
        "concurrent": concurrent,
        "model":      args.model,
        "temperature":args.temperature,
        "resume":     args.resume,
        "data_root":  data_root,
        "data_roots": data_roots,
        "out":        out,
        "stages":     stages,
        "dry_run":    args.dry_run,
    }


# ─────────────────────────────────────────────────────────────────
# Stage 1: Sweep
# ─────────────────────────────────────────────────────────────────

async def stage_sweep(cfg: dict) -> bool:
    from dotenv import load_dotenv
    load_dotenv()

    from loggers.schemas import TopologyName
    from execution.runner import SweepRunner, SweepConfig
    from benchmark_wrappers.task_curator import curate_tasks

    benchmarks = cfg["benchmarks"] or ALL_BENCHMARKS
    data_root  = cfg["data_root"]
    topologies = list(TopologyName)

    # Load tasks
    all_tasks = curate_tasks(data_root=data_root.parent, save_manifest=True)
    tasks = []
    for bench in benchmarks:
        bench_tasks = [t for t in all_tasks if t.benchmark == bench][:cfg["max_tasks"]]
        tasks.extend(bench_tasks)
        print(f"    {bench}: {len(bench_tasks)} tasks")

    if not tasks:
        print("  ERROR: no tasks loaded — check benchmark data paths")
        return False

    total_runs = len(tasks) * len(topologies) * len(cfg["scales"]) * len(cfg["seeds"])
    print(f"\n  Plan: {total_runs} runs")
    print(f"    Topologies:   {[t.value for t in topologies]}")
    print(f"    Scales:       {cfg['scales']}")
    print(f"    Seeds:        {cfg['seeds']}")
    print(f"    Tasks:        {len(tasks)}")
    print(f"    Max steps:    {cfg['max_steps']}")
    print(f"    Concurrent:   {cfg['concurrent']}")
    print(f"    Data root:    {data_root}")
    print(f"    Est. tokens:  ~{total_runs * 5000:,}")

    if cfg["dry_run"]:
        print("\n  [DRY RUN — no LLM calls]")
        return True

    # Build LLM
    from llm_factory import build_llm
    llm = build_llm(cfg["model"], cfg["temperature"])

    sweep_cfg = SweepConfig(
        topologies=topologies,
        agent_scales=cfg["scales"],
        seeds=cfg["seeds"],
        max_concurrent=cfg["concurrent"],
        skip_existing=cfg["resume"],
        snapshot_every=5,
        max_steps=cfg["max_steps"],
        model_name=cfg["model"],
        temperature=cfg["temperature"],
    )

    runner  = SweepRunner(llm=llm, data_root=data_root)
    results = await runner.run_sweep(tasks, sweep_cfg)

    n_ok    = sum(1 for r in results if r.success)
    n_fail  = len(results) - n_ok
    print(f"\n  Sweep done: {n_ok}/{len(results)} OK  ({n_fail} failed)")

    event_files = list(data_root.rglob("events.jsonl"))
    print(f"  Event files: {len(event_files)}")

    return len(event_files) > 0


# ─────────────────────────────────────────────────────────────────
# Stage 2: Extract
# ─────────────────────────────────────────────────────────────────

def stage_extract(cfg: dict) -> dict:
    from event_extraction.coordination import (
        _load_all_events, filter_events,
        extract_delegation_cascades, extract_revision_waves,
        extract_contradiction_bursts, extract_merge_fanin,
        extract_influence_per_agent,
    )
    from event_extraction.tce import tce_per_run
    from collections import defaultdict

    data_roots   = cfg["data_roots"]
    extracted    = cfg["out"] / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)

    extractors = {
        "delegation_cascade":  extract_delegation_cascades,
        "revision_wave":       extract_revision_waves,
        "contradiction_burst": extract_contradiction_bursts,
        "merge_fanin":         extract_merge_fanin,
        "influence_per_agent": extract_influence_per_agent,
    }

    # ── Pool + per-group observables ──────────────────────────────
    pooled: dict         = defaultdict(list)
    by_group: dict       = {}
    run_rows: list       = []
    event_count_total    = 0

    print(f"\n  Scanning {len(data_roots)} data root(s)...")
    for root in data_roots:
        if not root.exists():
            print(f"    SKIP (not found): {root}")
            continue

        bench = root.name
        all_events = _load_all_events(root)
        event_count_total += len(all_events)
        print(f"    {bench}: {len(all_events)} events")

        if not all_events:
            continue

        # Per-topology per-N groups
        groups: dict = defaultdict(list)
        for ev in all_events:
            topo = ev.get("topology", "unknown")
            n    = ev.get("num_agents", 0)
            groups[(topo, bench, n)].append(ev)

        # Pooled observables
        for name, fn in extractors.items():
            sizes = fn(all_events)
            pooled[name].extend(sizes)
        pooled["tce_per_run"].extend(tce_per_run(root))

        # Per-group
        for (topo, b, n), evs in groups.items():
            key = f"{topo}__{b}__n{n}"
            by_group[key] = {}
            for name, fn in extractors.items():
                by_group[key][name] = fn(evs)

        # H2 run metrics
        for meta_path in sorted(root.rglob("run_metadata.json")):
            try:
                meta  = json.loads(meta_path.read_text())
                run_id = meta.get("run_id", "")
                parts  = run_id.split("__")
                row = {
                    "run_id":     run_id,
                    "benchmark":  parts[0] if len(parts) > 0 else bench,
                    "topology":   parts[1] if len(parts) > 1 else "unknown",
                    "num_agents": int(parts[2].replace("n","")) if len(parts) > 2 and parts[2][1:].isdigit() else 0,
                    "seed":       parts[3].replace("s","") if len(parts) > 3 else "0",
                    "task_id":    parts[4] if len(parts) > 4 else "unknown",
                    # H2 core
                    "task_score":              meta.get("task_score"),
                    "completion_ratio":        meta.get("completion_ratio"),
                    "coherence_score":         meta.get("coherence_score"),
                    "integration_score":       meta.get("integration_score"),
                    "num_revisions_total":     meta.get("num_revisions_total"),
                    "num_contradictions_total":meta.get("num_contradictions_total"),
                    "num_merges_total":        meta.get("num_merges_total"),
                    "num_endorsements_total":  meta.get("num_endorsements_total"),
                    "num_claims_total":        meta.get("num_claims_total"),
                    "num_subtasks_total":      meta.get("num_subtasks_total"),
                    "tokens_total":            meta.get("tokens_total"),
                    "wall_time_seconds":       meta.get("wall_time_seconds"),
                    "success_per_token":       meta.get("success_per_token"),
                    "quality_adjusted_efficiency": meta.get("quality_adjusted_efficiency"),
                    # New H2 metrics from extra
                    "claim_participation_rate": meta.get("extra", {}).get("claim_participation_rate"),
                    "resolution_rate":          meta.get("extra", {}).get("resolution_rate"),
                    "revisions_per_claim":      meta.get("extra", {}).get("revisions_per_claim"),
                    "merges_per_claim":         meta.get("extra", {}).get("merges_per_claim"),
                    "architecture":             meta.get("extra", {}).get("architecture"),
                    # H3
                    "num_unique_agents_activated": meta.get("num_unique_agents_activated"),
                    "activation_rate":           meta.get("activation_rate"),
                    # Benchmark-specific
                    "gaia_exact_match":          meta.get("gaia_exact_match"),
                    "gaia_rubric_score":         meta.get("gaia_rubric_score"),
                    "swe_patch_applied":         meta.get("swe_patch_applied"),
                }
                run_rows.append(row)
            except Exception:
                pass

    print(f"\n  Total events: {event_count_total}")
    print(f"  Run records:  {len(run_rows)}")
    print("\n  H1 observable counts:")
    for name, sizes in sorted(pooled.items()):
        import numpy as np
        arr = [s for s in sizes if s > 0]
        xmax = float(max(arr)) if arr else 0
        print(f"    {name:<28} n={len(arr):6d}  xmax={xmax:.1f}")

    # Save pooled
    (extracted / "observables_pooled.json").write_text(
        json.dumps(dict(pooled), indent=2)
    )
    (extracted / "observables_by_group.json").write_text(
        json.dumps(by_group, indent=2)
    )

    # Save run_metrics.csv
    if run_rows:
        try:
            import pandas as pd  # type: ignore
            pd.DataFrame(run_rows).to_csv(extracted / "run_metrics.csv", index=False)
            print(f"\n  H2 run_metrics.csv: {len(run_rows)} rows")

            # Topology summary
            df = pd.DataFrame(run_rows)
            agg_cols = ["task_score","completion_ratio","coherence_score",
                        "integration_score","tokens_total","claim_participation_rate",
                        "resolution_rate"]
            agg_cols = [c for c in agg_cols if c in df.columns]
            summary = df.groupby("topology")[agg_cols].agg(["mean","std"]).round(4)
            summary.columns = ["__".join(c) for c in summary.columns]
            summary = summary.reset_index()
            summary.to_csv(extracted / "topology_summary.csv", index=False)
            print(f"  H2 topology_summary.csv:")
            for _, row in summary.iterrows():
                score = row.get("task_score__mean", "?")
                coh   = row.get("coherence_score__mean", "?")
                comp  = row.get("completion_ratio__mean", "?")
                print(f"    {row['topology']:<22} score={score:.3f}  "
                      f"coherence={coh:.3f}  completion={comp:.3f}")
        except ImportError:
            print("  (pandas not available — skipping CSV)")

    # H3 scaling table
    scaling_rows = []
    for group_key, obs_dict in by_group.items():
        parts = group_key.split("__")
        topo  = parts[0] if len(parts) > 0 else "?"
        bench = parts[1] if len(parts) > 1 else "?"
        n_str = parts[2].replace("n","") if len(parts) > 2 else "0"
        for obs_name, sizes in obs_dict.items():
            arr = [s for s in sizes if s > 0]
            if arr:
                import numpy as np
                scaling_rows.append({
                    "topology": topo, "benchmark": bench,
                    "num_agents": int(n_str) if n_str.isdigit() else 0,
                    "observable": obs_name,
                    "n": len(arr),
                    "mean": round(float(np.mean(arr)), 4),
                    "xmax": round(float(np.max(arr)), 2),
                    "gini": round(float(_gini(np.array(arr))), 4),
                    "p90":  round(float(np.percentile(arr, 90)), 4),
                })
    if scaling_rows:
        try:
            import pandas as pd  # type: ignore
            pd.DataFrame(scaling_rows).to_csv(extracted / "scaling_table.csv", index=False)
            print(f"  H3 scaling_table.csv: {len(scaling_rows)} rows")
        except ImportError:
            pass

    # ── Graph reconstruction ──────────────────────────────────────
    try:
        from event_extraction.graph_builder import (
            extract_graph_rows, graph_observables_from_tables,
        )
        claim_nodes, claim_edges, agent_edges, run_graph_summary = extract_graph_rows(
            data_roots,
            topology_filter=cfg.get("topology_filter"),
            benchmark_filter=cfg.get("benchmark_filter"),
            n_filter=cfg.get("n_filter"),
            verbose=False,
        )
        _write_csv_simple(claim_nodes,       extracted / "claim_nodes.csv")
        _write_csv_simple(claim_edges,       extracted / "claim_edges.csv")
        _write_csv_simple(agent_edges,       extracted / "agent_edges.csv")
        _write_csv_simple(run_graph_summary, extracted / "run_graph_summary.csv")

        # Add graph-derived observables into pooled pool
        graph_obs = graph_observables_from_tables(claim_nodes, claim_edges)
        print("\n  Graph-derived H1 observables:")
        for obs_name, sizes in sorted(graph_obs.items()):
            arr = [s for s in sizes if s > 0]
            if arr:
                pooled.setdefault(obs_name, []).extend(arr)
                import numpy as _np
                print(f"    {obs_name:<28} n={len(arr):6d}  xmax={float(max(arr)):.1f}")
        # Re-save pooled with graph additions
        (extracted / "observables_pooled.json").write_text(json.dumps(dict(pooled), indent=2))
    except Exception as e:
        print(f"  Graph reconstruction: {e} (non-fatal)")

    print(f"\n  Saved to {extracted}/")
    return dict(pooled)


def _write_csv_simple(rows: list, path: Path) -> None:
    """Write list of dicts to CSV without pandas dependency."""
    if not rows:
        return
    try:
        import pandas as pd  # type: ignore
        pd.DataFrame(rows).to_csv(path, index=False)
    except ImportError:
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  → {path.name}  ({len(rows)} rows)")


def _gini(arr):
    import numpy as np
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum()) - (n + 1) / n))


# ─────────────────────────────────────────────────────────────────
# Stage 3: Analyse (power-law fit)
# ─────────────────────────────────────────────────────────────────

def stage_analyse(cfg: dict, obs: dict):
    from tail_fitting.powerlaw_fit import fit_all, fit_observable, fits_to_dataframe, OBS_SPECS

    analysis = cfg["out"] / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    # Relax thresholds for smoke
    min_n = 10 if cfg["mode"] == "smoke" else 50

    print(f"\n  Fitting {len(obs)} observables (min_n={min_n})...")
    fits = fit_all(obs, specs=OBS_SPECS, verbose=True,
                   min_n_total=min_n, min_n_tail=min_n)

    n_pl  = sum(1 for r in fits.values() if r.pl_plausible)
    n_sup = sum(1 for r in fits.values() if r.pl_supported)
    n_aic = sum(1 for r in fits.values() if r.best_model_by_aic == "power_law")
    print(f"\n  Fitted: {len(fits)}/{len(obs)}  PL plausible: {n_pl}  "
          f"PL supported: {n_sup}  Best AIC: {n_aic}")

    # Save JSON — cast numpy scalars to Python natives for JSON compat
    import numpy as _np

    def _to_native(v):
        if isinstance(v, _np.bool_):    return bool(v)
        if isinstance(v, _np.integer):  return int(v)
        if isinstance(v, _np.floating): return None if _np.isnan(v) else float(v)
        return v

    fit_out = {}
    for k, r in fits.items():
        fit_out[k] = {fk: _to_native(fv) for fk, fv in {
            "alpha": r.alpha, "sigma_alpha": r.sigma_alpha,
            "x_min": r.x_min, "n_tail": r.n_tail,
            "tail_fraction": r.tail_fraction,
            "ks_statistic": r.ks_statistic,
            "comparison_vs_lognormal":    r.comparison_vs_lognormal,
            "comparison_vs_exponential":  r.comparison_vs_exponential,
            "comparison_vs_truncated_pl": r.comparison_vs_truncated_pl,
            "aic_powerlaw": r.aic_powerlaw,
            "aic_lognormal": r.aic_lognormal,
            "best_model_by_aic": r.best_model_by_aic,
            "ols_loglog_slope": r.ols_loglog_slope,
            "pl_plausible": r.pl_plausible,
            "pl_supported": r.pl_supported,
            "regime": r.regime,
            "quality_tag": r.quality_tag,
            "gini": r.gini,
        }.items()}
    (analysis / "fit_results.json").write_text(json.dumps(fit_out, indent=2))

    # Pooled CSV
    try:
        import pandas as _pd  # type: ignore
        df = fits_to_dataframe(fits)
        df.to_csv(analysis / "fit_results.csv", index=False)
        print(f"  Saved → {analysis}/fit_results.csv")
    except ImportError:
        _pd = None

    # ── Per-group fits for Tables 2 + 3 ──────────────────────────────────────
    # Merge event-native by_group observables with graph-derived per-topology observables
    by_group_path = cfg["out"] / "extracted" / "observables_by_group.json"
    if _pd is not None:
        try:
            # Start with event-native observables_by_group.json
            by_group: dict = {}
            if by_group_path.exists():
                by_group = json.loads(by_group_path.read_text())

            # Add graph-derived observables per topology from claim CSVs
            cn_path = cfg["out"] / "extracted" / "claim_nodes.csv"
            ce_path = cfg["out"] / "extracted" / "claim_edges.csv"
            if cn_path.exists():
                try:
                    from event_extraction.graph_builder import graph_observables_from_tables
                    cn = _pd.read_csv(cn_path).to_dict("records")
                    ce = _pd.read_csv(ce_path).to_dict("records") if ce_path.exists() else []

                    # Group claim nodes/edges by (topology, benchmark, num_agents)
                    from collections import defaultdict as _dd
                    g_nodes: dict = _dd(list)
                    g_edges: dict = _dd(list)
                    for n in cn:
                        k = (n.get("topology","unknown"),
                             n.get("benchmark","unknown"),
                             n.get("num_agents", 0))
                        g_nodes[k].append(n)
                    for e in ce:
                        k = (e.get("topology","unknown"),
                             e.get("benchmark","unknown"),
                             e.get("num_agents", 0))
                        g_edges[k].append(e)

                    for (topo, bench, n_a) in g_nodes:
                        key_str = f"{topo}__{bench}__n{n_a}"
                        g_obs = graph_observables_from_tables(
                            g_nodes[(topo, bench, n_a)],
                            g_edges.get((topo, bench, n_a), [])
                        )
                        if key_str not in by_group:
                            by_group[key_str] = {}
                        for obs_name, vals in g_obs.items():
                            by_group[key_str][obs_name] = vals
                    print(f"  Graph observables added to {len(by_group)} groups")
                except Exception as e:
                    print(f"  Graph per-group obs: {e} (non-fatal)")

            # Fit all groups
            group_rows = []
            print(f"\n  Fitting per-group slices ({len(by_group)} groups)...")
            for group_key, group_obs in by_group.items():
                parts    = group_key.split("__")
                topo     = parts[0] if len(parts) > 0 else "unknown"
                bench    = parts[1] if len(parts) > 1 else "unknown"
                n_str    = parts[2].replace("n", "") if len(parts) > 2 else "0"
                n_agents = int(n_str) if n_str.isdigit() else 0

                for obs_name, sizes in group_obs.items():
                    arr = [s for s in sizes if s and s > 0]
                    if len(arr) < min_n:
                        continue
                    spec = OBS_SPECS.get(obs_name, {"discrete": True})
                    r = fit_observable(
                        arr, obs_name,
                        discrete=spec.get("discrete", True),
                        verbose=False,
                        min_n_total=min_n, min_n_tail=min_n,
                    )
                    if r is None:
                        continue
                    group_rows.append({
                        "observable":                obs_name,
                        "topology":                  topo,
                        "benchmark":                 bench,
                        "num_agents":                n_agents,
                        "n_tail":                    r.n_tail,
                        "tail_fraction":             r.tail_fraction,
                        "alpha":                     r.alpha,
                        "sigma_alpha":               r.sigma_alpha,
                        "x_min":                     r.x_min,
                        "ks_statistic":              r.ks_statistic,
                        "comparison_vs_lognormal":   r.comparison_vs_lognormal,
                        "comparison_vs_exponential": r.comparison_vs_exponential,
                        "comparison_vs_truncated_pl":r.comparison_vs_truncated_pl,
                        "best_model_by_aic":         r.best_model_by_aic,
                        "ols_loglog_slope":          r.ols_loglog_slope,
                        "pl_plausible":              bool(r.pl_plausible),
                        "pl_supported":              bool(r.pl_supported),
                        "regime":                    r.regime,
                        "quality_tag":               r.quality_tag,
                        "gini":                      r.gini,
                    })

            if group_rows:
                _pd.DataFrame(group_rows).to_csv(analysis / "fits_by_group.csv", index=False)
                print(f"  Saved → {analysis}/fits_by_group.csv  ({len(group_rows)} group fits)")
            else:
                print("  fits_by_group.csv: no group fits passed thresholds "
                      f"(min_n={min_n} — expected for smoke, fine for full sweep)")
        except Exception as e:
            print(f"  fits_by_group: {e} (non-fatal)")
    else:
        print("  fits_by_group.csv: skipped (pandas unavailable)")

    return fits


# ─────────────────────────────────────────────────────────────────
# Stage 4: Tables
# ─────────────────────────────────────────────────────────────────

def stage_tables(cfg: dict):
    from tables.generate_tables import generate_all_tables

    tables_dir   = cfg["out"] / "tables"
    analysis_dir = cfg["out"] / "analysis"
    extracted    = cfg["out"] / "extracted"

    generate_all_tables(
        data_roots=cfg["data_roots"],
        out_dir=tables_dir,
        analysis_dir=analysis_dir if analysis_dir.exists() else None,
        extracted_dir=extracted   if extracted.exists()    else None,
    )
    print(f"  Tables → {tables_dir}/")


# ─────────────────────────────────────────────────────────────────
# Stage 5: Figures
# ─────────────────────────────────────────────────────────────────

def stage_figures(cfg: dict):
    import matplotlib
    matplotlib.use("Agg")

    figures_dir = cfg["out"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roots = cfg["data_roots"]

    # For smoke, skip figures that need snapshots.jsonl (not generated in smoke)
    if cfg["mode"] == "smoke":
        skip_figs = {"entropy_vs_alpha", "path_length_vs_alpha"}
    else:
        skip_figs = set()

    FIGURE_REGISTRY = [
        # (fig_id, module, function, extra_kwargs)
        ("ccdf_panel",            "visualization.ccdf_panel",            "plot_ccdf_panel",            {}),
        ("powerlaw_fit",          "visualization.powerlaw_fit_plot",      "plot_fit_comparison",        {}),
        ("lr_heatmap",            "visualization.lr_test_heatmap",        "plot_lr_heatmap",            {}),
        ("ccdf_by_family",        "visualization.ccdf_by_family",         "plot_ccdf_by_family",        {}),
        ("topology_comparison",   "visualization.topology_comparison",    "plot_topology_comparison",   {}),
        ("universality_collapse", "visualization.universality_collapse",  "plot_universality_collapse", {}),
        ("regime_phase",          "visualization.regime_phase_diagram",   "plot_regime_phase_diagram",  {}),
        ("lorenz_curves",         "visualization.lorenz_curves",          "plot_lorenz_curves",         {}),
        ("event_vs_success",      "visualization.event_vs_success",       "plot_event_vs_success",      {}),
        ("cognitive_work",        "visualization.cognitive_work_scaling", "plot_cognitive_work_scaling",{}),
        ("coalition_sankey",      "visualization.coalition_sankey",       "plot_coalition_sankey",      {}),
        ("coalition_timeline",    "visualization.coalition_timeline",     "plot_coalition_timeline",    {}),
        ("error_correction",      "visualization.error_correction_avalanche","plot_error_correction_avalanche",{}),
        ("avalanche_size_duration","visualization.avalanche_size_duration","plot_avalanche_size_duration",{}),
        ("avalanche_heatmap",     "visualization.avalanche_raster",       "plot_avalanche_heatmap",     {}),
        ("temporal_burst",        "visualization.temporal_burst_plot",    "plot_raster_for_best_run",   {}),
        ("entropy_vs_alpha",      "visualization.entropy_vs_alpha",       "plot_entropy_vs_alpha",      {}),
        ("path_length_vs_alpha",  "visualization.path_length_vs_alpha",   "plot_path_length_vs_alpha",  {}),
        ("mega_cascade",          "visualization.mega_cascade",           "plot_mega_cascade",          {}),
        ("xmax_scaling",          "visualization.xmax_scaling",           "plot_xmax_scaling",          {}),
        ("mean_vs_max_scaling",   "visualization.mean_vs_max_scaling",    "plot_mean_vs_max_scaling",   {}),
    ]

    n_ok = n_skip = n_fail = 0
    for fig_id, mod_name, fn_name, extra in FIGURE_REGISTRY:
        if fig_id in skip_figs:
            print(f"  –  {fig_id}  (skipped — needs snapshots.jsonl)")
            n_skip += 1
            continue
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            fn  = getattr(mod, fn_name)
            fn(data_roots=roots, out_dir=figures_dir, **extra)
            n_ok += 1
        except Exception as e:
            print(f"  ✗  {fig_id}: {e}")
            n_fail += 1

    print(f"\n  Figures: {n_ok} OK  {n_skip} skipped  {n_fail} failed → {figures_dir}/")


# ─────────────────────────────────────────────────────────────────
# Stage runner with timing + error isolation
# ─────────────────────────────────────────────────────────────────

stage_results: dict = {}


def run_stage(name: str, fn, *args, **kwargs):
    print(f"\n{'─'*60}")
    print(f"Stage: {name.upper()}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        stage_results[name] = ("✓", elapsed, None)
        print(f"\n  ✓  {name} done  ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        stage_results[name] = ("✗", elapsed, str(e))
        print(f"\n  ✗  {name} FAILED  ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()
    cfg  = resolve_config(args)

    print(f"\n{'='*60}")
    print(f"MAS Power-Laws Pipeline  [{cfg['mode'].upper()}]")
    print(f"{'='*60}")
    print(f"  Benchmarks:   {cfg['benchmarks'] or 'all'}")
    print(f"  Scales:       {cfg['scales']}")
    print(f"  Seeds:        {cfg['seeds']}")
    print(f"  Max tasks:    {cfg['max_tasks']}")
    print(f"  Data root:    {cfg['data_root']}")
    print(f"  Output:       {cfg['out']}")
    print(f"  Stages:       {cfg['stages']}")
    print(f"  Model:        {cfg['model']}")

    obs   = {}
    fits  = {}

    # Stage 1: Sweep
    if "sweep" in cfg["stages"]:
        ok = await run_stage("sweep", stage_sweep, cfg)
        if ok is False and not cfg["dry_run"]:
            print("\nERROR: sweep produced no data — aborting")
            sys.exit(1)

    # Stage 2: Extract
    if "extract" in cfg["stages"]:
        result = run_stage("extract", stage_extract, cfg)
        if result is not None:
            obs = result
        else:
            # Try loading from disk
            pooled_path = cfg["out"] / "extracted" / "observables_pooled.json"
            if pooled_path.exists():
                obs = json.loads(pooled_path.read_text())

    # Stage 3: Analyse
    # Always try to load obs from disk if not populated by extract stage
    if "analyse" in cfg["stages"]:
        if not obs:
            pooled_path = cfg["out"] / "extracted" / "observables_pooled.json"
            if pooled_path.exists():
                obs = json.loads(pooled_path.read_text())
                print(f"  Loaded observables from {pooled_path} ({len(obs)} observables)")
            else:
                print(f"  WARN: no observables_pooled.json found at {pooled_path}")
                print(f"  Run --stages extract first, or include extract in stages")
        if obs:
            result = run_stage("analyse", stage_analyse, cfg, obs)
            if result is not None:
                fits = result
        else:
            print("  analyse: skipped — no observables loaded")

    # Stage 4: Tables
    if "tables" in cfg["stages"]:
        run_stage("tables", stage_tables, cfg)

    # Stage 5: Figures
    if "figures" in cfg["stages"]:
        run_stage("figures", stage_figures, cfg)

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    total_time = sum(t for _, t, _ in stage_results.values())
    for stage, (status, elapsed, err) in stage_results.items():
        suffix = f"  → {err}" if err else ""
        print(f"  {status}  {stage:<12} {elapsed:6.1f}s{suffix}")
    print(f"{'─'*60}")
    print(f"  Total: {total_time:.1f}s")

    n_fail = sum(1 for s, _, _ in stage_results.values() if s == "✗")
    if n_fail == 0:
        print(f"\n✓ ALL STAGES PASSED")
        if cfg["mode"] == "smoke":
            print("\nSmoke test passed. Ready for full sweep:")
            print("  python main.py --mode full --benchmark MARBLE \\")
            print("      --data-root /scratch/kavanav/mas-results/MARBLE")
    else:
        print(f"\n✗ {n_fail} stage(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())