#!/usr/bin/env python3
"""
scripts/run_swe_eval.py
------------------------
Run the official SWE-bench Docker evaluation harness on predictions
collected during the sweep, then write results back into each run's
run_metadata.json.

Run this AFTER the full sweep completes:

    python scripts/run_swe_eval.py \
        --data-root /scratch/results/SWE-bench \
        --run-id my_sweep_001

Requirements:
    pip install swebench
    Docker must be running with sufficient disk space (~2GB per instance)

What it does:
    1. Reads data_root/swe_predictions.jsonl  (written by graph_runner.py)
    2. Calls swebench.harness.run_evaluation
    3. Parses the results JSON
    4. Patches swe_tests_passed, task_success, task_score into each
       run_metadata.json under data_root
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load_predictions(data_root: Path) -> list:
    preds_path = data_root / "swe_predictions.jsonl"
    if not preds_path.exists():
        print(f"No predictions file found at {preds_path}")
        sys.exit(1)
    preds = []
    for line in open(preds_path):
        line = line.strip()
        if line:
            preds.append(json.loads(line))
    print(f"Loaded {len(preds)} SWE-bench predictions")
    return preds


def run_harness(preds_path: Path, run_id: str, max_workers: int) -> Path:
    """
    Call swebench.harness.run_evaluation as a subprocess.
    Returns path to the results JSON.
    """
    results_dir = preds_path.parent / f"swe_harness_results_{run_id}"
    results_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name",    "princeton-nlp/SWE-bench_Verified",
        "--predictions_path", str(preds_path),
        "--max_workers",      str(max_workers),
        "--run_id",           run_id,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("SWE-bench harness exited with errors — check Docker and disk space")
        sys.exit(result.returncode)

    # Harness writes results to logs/run_evaluation/{run_id}/results.json
    results_path = Path("logs") / "run_evaluation" / run_id / "results.json"
    if not results_path.exists():
        # Try alternate location
        for candidate in Path("logs").rglob("results.json"):
            results_path = candidate
            break
    return results_path


def patch_run_metadata(data_root: Path, results: dict) -> None:
    """
    For each instance in results, find its run_metadata.json and
    update swe_tests_passed, task_success, task_score.

    results format (from swebench harness):
      { instance_id: { "resolved": bool, "tests_status": {...} }, ... }
    """
    patched = 0
    for instance_id, outcome in results.items():
        resolved = outcome.get("resolved", False)

        # Find all run dirs for this instance (may span multiple topologies/seeds)
        for meta_path in data_root.rglob(f"{instance_id}/run_metadata.json"):
            try:
                meta = json.loads(meta_path.read_text())
                meta["task_success"]    = resolved
                meta["task_score"]      = 1.0 if resolved else 0.0
                meta["swe_tests_passed"] = 1 if resolved else 0
                # Extract FAIL_TO_PASS count from harness output if available
                ts = outcome.get("tests_status", {})
                ftp_pass = ts.get("FAIL_TO_PASS", {}).get("success", [])
                ftp_fail = ts.get("FAIL_TO_PASS", {}).get("failure", [])
                meta["swe_tests_passed"] = len(ftp_pass)
                meta["swe_tests_total"]  = len(ftp_pass) + len(ftp_fail)
                meta_path.write_text(json.dumps(meta, indent=2))
                patched += 1
            except Exception as e:
                print(f"  Could not patch {meta_path}: {e}")

    print(f"Patched {patched} run_metadata.json files with SWE-bench results")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench Docker eval post-sweep")
    parser.add_argument("--data-root",   required=True, type=Path,
                        help="data_root used for the SWE-bench sweep")
    parser.add_argument("--run-id",      required=True,
                        help="Unique ID for this evaluation run")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel Docker workers (each needs ~2GB disk)")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Write deduplicated predictions file (take last prediction per instance)
    preds = load_predictions(data_root)
    seen  = {}
    for p in preds:
        seen[p["instance_id"]] = p   # last write wins per instance
    deduped_path = data_root / f"swe_predictions_deduped_{args.run_id}.jsonl"
    with open(deduped_path, "w") as f:
        for p in seen.values():
            f.write(json.dumps(p) + "\n")
    print(f"Deduped predictions: {len(seen)} instances → {deduped_path}")

    results_path = run_harness(deduped_path, args.run_id, args.max_workers)

    if results_path.exists():
        results = json.loads(results_path.read_text())
        resolved = sum(1 for v in results.values() if v.get("resolved"))
        total    = len(results)
        print(f"\nSWE-bench results: {resolved}/{total} resolved ({100*resolved/total:.1f}%)")
        patch_run_metadata(data_root, results)
    else:
        print(f"Results file not found at {results_path}")
        print("Check logs/run_evaluation/ manually")


if __name__ == "__main__":
    main()