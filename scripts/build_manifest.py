#!/usr/bin/env python3
"""
scripts/build_manifest.py
--------------------------
Loads all 4 benchmarks, selects the 70-task portfolio,
and saves data/task_manifest.json.

Run this ONCE before the sweep to verify your benchmark
connections and lock in the task IDs.

Usage
-----
    python scripts/build_manifest.py
    python scripts/build_manifest.py --data-root data --seed 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_wrappers.task_curator import curate_tasks, portfolio_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data", help="Where to save the manifest")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--skip",      nargs="+",
                   choices=["MARBLE","GAIA","REALM","SWE-bench"],
                   default=[],
                   help="Benchmarks to skip (if not yet available)")
    return p.parse_args()


def main():
    args  = parse_args()
    tasks = curate_tasks(
        data_root=Path(args.data_root),
        seed=args.seed,
        save_manifest=True,
    )
    print()
    print(portfolio_summary(tasks))
    print()
    print("Task families:")
    from collections import Counter
    families = Counter(t.task_family for t in tasks)
    for fam, count in sorted(families.items()):
        print(f"  {fam:<14}  {count:3d}")

    print(f"\nManifest → {Path(args.data_root) / 'task_manifest.json'}")
    print("Ready to run:  python scripts/run_sweep.py --dry-run --all --all-topologies")


if __name__ == "__main__":
    main()
