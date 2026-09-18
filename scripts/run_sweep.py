#!/usr/bin/env python3
"""
scripts/run_sweep.py
---------------------
CLI entry point for the MAS power-law sweep.

FULL SWEEP (4 GPUs, one benchmark per GPU)
------------------------------------------
  # GPU 0
  python scripts/run_sweep.py --benchmark MARBLE \\
      --data-root /scratch/kavanav/mas-results/MARBLE \\
      --all-topologies --scales 8 16 32 64 128 --seeds 0 1 2 \\
      --max-tasks 10 --max-concurrent 4

  # GPU 1
  python scripts/run_sweep.py --benchmark GAIA \\
      --data-root /scratch/kavanav/mas-results/GAIA \\
      --all-topologies --scales 8 16 32 64 128 --seeds 0 1 2 \\
      --max-tasks 10 --max-concurrent 4

  # GPU 2
  python scripts/run_sweep.py --benchmark REALM \\
      --data-root /scratch/kavanav/mas-results/REALM \\
      --all-topologies --scales 8 16 32 64 128 --seeds 0 1 2 \\
      --max-tasks 10 --max-concurrent 4

  # GPU 3
  python scripts/run_sweep.py --benchmark SWE-bench \\
      --data-root /scratch/kavanav/mas-results/SWE-bench \\
      --all-topologies --scales 8 16 32 64 128 --seeds 0 1 2 \\
      --max-tasks 10 --max-concurrent 4

SMOKE TEST
----------
  python scripts/run_sweep.py --benchmark MARBLE \\
      --topology chain tree \\
      --scales 8 --seeds 0 \\
      --max-tasks 5 --data-root data/smoke \\
      --max-concurrent 4

RESUME PARTIAL RUN
------------------
  python scripts/run_sweep.py --benchmark MARBLE \\
      --data-root /scratch/.../MARBLE --all-topologies \\
      --scales 8 16 32 64 128 --seeds 0 1 2 --resume

POST-SWEEP ANALYSIS
-------------------
  python scripts/extract_events.py \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA \\
                   /scratch/.../REALM  /scratch/.../SWE-bench \\
      --out data/extracted

  python scripts/run_analysis.py \\
      --observables data/extracted/observables_pooled.json \\
      --by-group    data/extracted/observables_by_group.json \\
      --out data/analysis

  python scripts/generate_tables.py \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA ... \\
      --analysis data/analysis --extracted data/extracted --out tables/
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loggers.schemas import TopologyName
from execution.runner import SweepRunner, SweepConfig
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAS power-law sweep runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Benchmark
    bench_group = p.add_mutually_exclusive_group()
    bench_group.add_argument(
        "--benchmark", "-b", nargs="+",
        choices=["MARBLE", "GAIA", "REALM", "SWE-bench"],
        metavar="BENCH",
        help="Benchmark(s) to run. For multi-GPU: one benchmark per GPU.",
    )
    bench_group.add_argument(
        "--all-benchmarks", action="store_true",
        help="Run all 4 benchmarks sequentially (single GPU only)",
    )

    # Topology
    topo_group = p.add_mutually_exclusive_group()
    topo_group.add_argument(
        "--topology", "-t", nargs="+",
        choices=[t.value for t in TopologyName],
        metavar="TOPO",
        help="Topology/topologies to run",
    )
    topo_group.add_argument(
        "--all-topologies", action="store_true",
        help="Run all 7 topologies",
    )

    # Scale / seeds
    p.add_argument("--scales", "--n", type=int, nargs="+", default=[8],
                   metavar="N", dest="scales",
                   help="Agent counts  e.g. --scales 8 16 32 64 128")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   metavar="S",
                   help="Random seeds  e.g. --seeds 0 1 2")

    # Task selection
    p.add_argument("--max-tasks", type=int, default=10, dest="max_tasks",
                   help="Max tasks per benchmark (default 10 for full sweep)")
    p.add_argument("--manifest", default=None, metavar="PATH",
                   help="Path to task_manifest.json (reproducible task set)")

    # Output
    p.add_argument("--data-root", default="data/runs", metavar="PATH",
                   help=(
                       "Root directory for this benchmark's run artefacts.\n"
                       "Use SEPARATE paths per benchmark on multi-GPU setups.\n"
                       "Layout: {data-root}/{topology}/n{N}/s{seed}/{task_id}/"
                   ))

    # Model
    p.add_argument("--model",          default="gpt-4o-mini")
    p.add_argument("--temperature",    type=float, default=0.7)

    # Execution
    p.add_argument("--max-concurrent", type=int, default=4, dest="max_concurrent",
                   help="Concurrent runs (default 4; lower for large N)")
    p.add_argument("--max-steps",      type=int, default=20, dest="max_steps")
    p.add_argument("--snapshot-every", type=int, default=5,  dest="snapshot_every")
    p.add_argument("--resume",         action="store_true",
                   help="Skip runs where run_metadata.json already exists")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print job plan without running")
    p.add_argument("--dti",            action="store_true",
                   help="Enable Deficit-Triggered Integration (see src/dti/dti.py)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────

from llm_factory import build_llm


# ─────────────────────────────────────────────────────────────────
# Task loading
# ─────────────────────────────────────────────────────────────────

def load_tasks(args):
    if args.all_benchmarks:
        bench_filter = None
    elif args.benchmark:
        bench_filter = set(args.benchmark)
    else:
        print("ERROR: specify --benchmark MARBLE (or --all-benchmarks)")
        sys.exit(1)

    print(f"Loading tasks for: {bench_filter or 'all benchmarks'} ...")

    if args.manifest and Path(args.manifest).exists():
        from benchmark_wrappers.task_curator import load_from_manifest
        tasks = load_from_manifest(
            Path(args.manifest),
            benchmarks=list(bench_filter) if bench_filter else None,
        )
        print(f"  Loaded {len(tasks)} tasks from manifest")
    else:
        from benchmark_wrappers.task_curator import curate_tasks
        tasks = curate_tasks(
            data_root=Path(args.data_root).parent,
            save_manifest=True,
        )
        if bench_filter:
            tasks = [t for t in tasks if t.benchmark in bench_filter]

    # Apply max-tasks cap per benchmark
    if args.max_tasks:
        from collections import defaultdict
        by_bench: dict = defaultdict(list)
        for t in tasks:
            by_bench[t.benchmark].append(t)
        tasks = []
        for bench, bench_tasks in by_bench.items():
            tasks.extend(bench_tasks[:args.max_tasks])
            print(f"  {bench}: {min(len(bench_tasks), args.max_tasks)} tasks")

    print(f"  Total: {len(tasks)} tasks")
    if not tasks:
        print("ERROR: no tasks loaded.")
        sys.exit(1)
    return tasks


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    # Topologies
    if args.all_topologies:
        topologies = list(TopologyName)
    elif args.topology:
        topologies = [TopologyName(t) for t in args.topology]
    else:
        print("ERROR: specify --topology chain (or --all-topologies)")
        sys.exit(1)

    tasks = load_tasks(args)

    config = SweepConfig(
        topologies=topologies,
        agent_scales=args.scales,
        seeds=args.seeds,
        max_concurrent=args.max_concurrent,
        skip_existing=args.resume,
        snapshot_every=args.snapshot_every,
        max_steps=args.max_steps,
        model_name=args.model,
        temperature=args.temperature,
        dti_enabled=args.dti,
    )

    # Dry run
    if args.dry_run:
        total = len(tasks) * len(topologies) * len(args.scales) * len(args.seeds)
        benches = sorted({t.benchmark for t in tasks})
        print(f"\nDRY RUN — {total} jobs planned")
        print(f"  Benchmarks:   {benches}")
        print(f"  Topologies:   {[t.value for t in topologies]}")
        print(f"  Scales:       {args.scales}")
        print(f"  Seeds:        {args.seeds}")
        print(f"  Tasks:        {len(tasks)}")
        print(f"  Max tasks/bench: {args.max_tasks}")
        print(f"  Data root:    {args.data_root}")
        print(f"  Resume:       {args.resume}")
        print(f"  Max concurrent: {args.max_concurrent}")
        print(f"\nToken estimate: ~{total * 5000:,} tokens  "
              f"(~${total * 5000 * 0.15 / 1_000_000:.2f} at $0.15/1M)")
        print(f"\nSample paths:")
        for task in tasks[:2]:
            for topo in topologies[:2]:
                path = (
                    Path(args.data_root) / topo.value
                    / f"n{args.scales[0]}" / f"s{args.seeds[0]}" / task.task_id
                )
                print(f"    {path}/")
        return

    # Run
    llm     = build_llm(args.model, args.temperature)
    runner  = SweepRunner(llm=llm, data_root=Path(args.data_root))
    results = await runner.run_sweep(tasks, config)
    print(f"\nDone. {len(results)} runs completed.")
    print(f"Results in: {args.data_root}")
    print(f"\nNext steps:")
    print(f"  python scripts/extract_events.py --data-roots {args.data_root} --out data/extracted")
    print(f"  python scripts/run_analysis.py --observables data/extracted/observables_pooled.json --out data/analysis")


if __name__ == "__main__":
    asyncio.run(main())