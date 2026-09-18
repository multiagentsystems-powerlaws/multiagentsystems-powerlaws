"""
src/analysis/run_pipeline.py

End-to-end post-hoc pipeline: raw events.jsonl → power-law fit results.

This is the single wiring function that connects all new modules.
Called by scripts/fit_tails.py and scripts/run_sweep.py after execution.

Pipeline stages:
  1. Load    read events.jsonl for one run directory
  2. Annotate  event_extractor.annotate_event_types()
  3. Build DAG  dag_builder.build_all()
  4. Extract   cascade_metrics.extract_all_observables()
  5. Fit       tail_fitting.powerlaw_fit.fit_all()  (on pooled data)

Stages 1-4 run per-run. Stage 5 runs once on the pooled sample across
all runs in a condition (same topology + task_family + N + benchmark).

Usage (single run, for testing):
    from analysis.run_pipeline import process_run, pool_and_fit

    obs = process_run(Path("data/runs/gaia/chain/n16/s0/task_001"))
    print(obs["event_observables"]["tce"])

Usage (full sweep):
    from analysis.run_pipeline import process_condition

    fits = process_condition(
        run_dirs=[...],
        condition_label="gaia_chain_n64",
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from event_extraction.event_extractor import annotate_event_types
from event_extraction.llm_event_classifier import annotate_event_types_llm
from event_extraction.resolve_event_labels import resolve_event_labels
from observables.dag_builder import build_all
from observables.cascade_metrics import extract_all_observables
from tail_fitting.powerlaw_fit import fit_all, FitResult


# ── Stage 1: Load ─────────────────────────────────────────────

def load_trace_rows(run_dir: Path) -> List[dict]:
    """
    Load all TraceRow dicts from a run directory's events.jsonl.
    Returns empty list (with warning) if file missing or malformed.
    """
    events_file = run_dir / "events.jsonl"
    if not events_file.exists():
        print(f"  [warn] events.jsonl not found: {run_dir}")
        return []

    rows = []
    with open(events_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] bad JSON at line {i+1} in {events_file}: {e}")
    return rows


# ── Stages 2-4: Annotate → DAG → Observables ─────────────────

def process_run(run_dir: Path, llm: object = None) -> dict:
    """
    Run the full post-hoc pipeline for a single run directory.

    Returns the output of cascade_metrics.extract_all_observables():
      {
        "event_observables": {
            "delegation_sizes":     List[int],
            "revision_waves":       List[int],
            "contradiction_bursts": List[int],
            "merge_fan_in":         List[int],
            "tce":                  List[int],
        },
        "agent_metrics": {agent_id: {...}, ...}
      }

    Also writes root_claim_id, claim_depth, subtask_depth back into
    each TraceRow via dag_builder (in-memory only, not persisted here).
    """
    rows = load_trace_rows(run_dir)
    if not rows:
        return {"event_observables": {k: [] for k in [
            "delegation_sizes", "revision_waves", "contradiction_bursts",
            "merge_fan_in", "tce"
        ]}, "agent_metrics": {}}

    # Stage 2a: rule-based annotation (structural + obvious semantic cases)
    rows = annotate_event_types(rows)

    # Stage 2b: LLM semantic disambiguation (only if llm provided)
    # Only processes rows flagged needs_llm_event_classification=True.
    # Safe to skip: rule labels remain as final labels in that case.
    if llm is not None:
        n_candidates = sum(1 for r in rows if r.get("needs_llm_event_classification"))
        if n_candidates == 0:
            import warnings
            warnings.warn(
                f"annotate_event_types_llm called but no rows are flagged "
                f"needs_llm_event_classification in {run_dir}. "
                "Check upstream annotation logic.",
                stacklevel=2,
            )
        rows = annotate_event_types_llm(rows, llm)

    # Stage 2c: resolve final labels from rule + LLM outputs
    rows = resolve_event_labels(rows)

    # Stage 3: build subtask tree, claim DAG, cascades
    # (also writes root_claim_id, claim_depth, subtask_depth back into rows)
    subtask_tree, claim_dag, cascades = build_all(rows)

    # Stage 4: extract observables
    return extract_all_observables(rows, subtask_tree, cascades)


# ── Stage 5: Pool and fit ─────────────────────────────────────

def pool_and_fit(
    per_run_observables: List[dict],
    condition_label: str = "",
    verbose: bool = True,
) -> Dict[str, FitResult]:
    """
    Pool event_observables across all runs in a condition, then fit.

    Args:
        per_run_observables: list of process_run() outputs
        condition_label:     printed in verbose output
        verbose:             print fit summary

    Returns:
        dict of observable_name -> FitResult
    """
    pooled: Dict[str, List[int]] = {
        "delegation_sizes":     [],
        "revision_waves":       [],
        "contradiction_bursts": [],
        "merge_fan_in":         [],
        "tce":                  [],
    }

    for obs in per_run_observables:
        eo = obs.get("event_observables", {})
        for key in pooled:
            pooled[key].extend(eo.get(key, []))

    if verbose and condition_label:
        print(f"\nCondition: {condition_label}")
        for key, vals in pooled.items():
            print(f"  {key}: n={len(vals)}"
                  + (f"  max={max(vals)}" if vals else "  (empty)"))

    return fit_all(pooled, verbose=verbose)


# ── Full condition pipeline ───────────────────────────────────

def process_condition(
    run_dirs: List[Path],
    condition_label: str = "",
    llm: object = None,
    verbose: bool = True,
) -> Dict[str, FitResult]:
    """
    Process all run directories for one experimental condition and fit.

    A condition = one (benchmark, topology, task_family, N) cell.
    run_dirs contains one entry per seed.

    Returns:
        dict of observable_name -> FitResult, one row per observable
    """
    t0 = time.time()
    per_run = []

    for run_dir in run_dirs:
        obs = process_run(run_dir, llm=llm)
        per_run.append(obs)

    fits = pool_and_fit(per_run, condition_label=condition_label, verbose=verbose)

    if verbose:
        elapsed = time.time() - t0
        print(f"  [{condition_label}] {len(run_dirs)} runs processed in {elapsed:.1f}s")

    return fits


# ── Sweep helper ──────────────────────────────────────────────

def discover_run_dirs(data_root: Path) -> Dict[str, List[Path]]:
    """
    Walk data_root and group run directories by condition key.

    Expected directory structure:
      data_root / benchmark / topology / n{N} / s{seed} / task_id /

    Returns:
      dict of condition_key -> list of run_dirs
      where condition_key = "{benchmark}_{topology}_n{N}"
    """
    conditions: Dict[str, List[Path]] = {}

    for events_file in sorted(data_root.rglob("events.jsonl")):
        run_dir = events_file.parent
        parts   = run_dir.relative_to(data_root).parts
        # parts: (benchmark, topology, nN, sSeed, task_id)
        if len(parts) < 3:
            continue
        benchmark = parts[0]
        topology  = parts[1]
        n_str     = parts[2]  # e.g. "n64"
        key       = f"{benchmark}_{topology}_{n_str}"
        conditions.setdefault(key, []).append(run_dir)

    return conditions