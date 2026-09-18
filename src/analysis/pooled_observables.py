"""
src/analysis/pooled_observables.py

Shared, correctness-verified H1 observable extraction, pooled across an arbitrary
set of data roots. This is the single implementation used by both
scripts/extract_events.py and scripts/main.py's stage_extract — previously
each had its own independent (and, for delegation/revision/merge/TCE,
incorrect) copy of this logic.

Pipeline per run: event_extractor.annotate_event_types -> resolve_event_labels
-> observables.dag_builder.build_all -> observables.cascade_metrics.extract_all_observables.
Building the claim DAG per-run (not on one global flat event list) is
required: claim_id and subtask_id are not guaranteed unique across different
runs, so pooling raw events before DAG construction would corrupt cascade
reconstruction.

influence_per_agent remains the exploratory, topology-specific proxy from
event_extraction/coordination.py (explicitly marked there as not a primary
observable) and is computed on the flat filtered event list, as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from event_extraction.coordination import (
    _load_all_events, _group_by_run, filter_events, extract_influence_per_agent,
)
from event_extraction.event_extractor import annotate_event_types
from event_extraction.resolve_event_labels import resolve_event_labels
from observables.dag_builder import build_all
from observables.cascade_metrics import extract_all_observables

# Maps cascade_metrics.py's observable names to the legacy key names the
# pipeline has always emitted downstream (tables/visualization).
OBSERVABLE_KEY_MAP = {
    "delegation_sizes":     "delegation_cascade",
    "revision_waves":       "revision_wave",
    "contradiction_bursts": "contradiction_burst",
    "merge_fan_in":         "merge_fanin",
    "tce":                  "tce_per_run",
}
CORE_OBSERVABLE_KEYS = list(OBSERVABLE_KEY_MAP.values())


def extract_run_observables(rows: List[Dict]) -> Dict[str, List[float]]:
    """
    Run the correctness-verified per-run pipeline on one run's rows, returning
    results under the legacy key names used throughout the pipeline.
    """
    rows = annotate_event_types(rows)
    rows = resolve_event_labels(rows)
    subtask_tree, claim_dag, cascades = build_all(rows)
    obs = extract_all_observables(rows, subtask_tree, cascades)
    eo = obs["event_observables"]
    return {legacy: eo[new] for new, legacy in OBSERVABLE_KEY_MAP.items()}


def pool_h1_observables(
    roots: List[Path],
    topology_filter: Optional[str] = None,
    benchmark_filter: Optional[str] = None,
    n_filter: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Extract H1 observables at three granularities from a set of data roots:
      - pooled (all runs merged)
      - by_topo  (per topology)
      - by_group (per topology, benchmark, N)

    Returns {"pooled": {...}, "by_topo": {...}, "by_group": {...}}, each a
    dict of observable_name -> List[float], using the legacy key names in
    CORE_OBSERVABLE_KEYS plus "influence_per_agent".
    """
    from collections import defaultdict

    all_events_flat: List[Dict] = []
    per_run_rows: Dict[str, List[Dict]] = {}
    run_group_key: Dict[str, tuple] = {}

    for root in roots:
        if not root.exists():
            continue
        all_evs = _load_all_events(root)
        filtered = filter_events(
            all_evs,
            topology=topology_filter,
            benchmark=benchmark_filter,
            num_agents=n_filter,
        )
        all_events_flat.extend(filtered)
        for run_id, rows in _group_by_run(filtered).items():
            per_run_rows.setdefault(run_id, []).extend(rows)
            first = rows[0]
            run_group_key[run_id] = (
                first.get("topology", "unknown"),
                first.get("benchmark", "unknown"),
                first.get("num_agents", 0),
            )

    result: Dict[str, Any] = {
        "pooled":   {k: [] for k in CORE_OBSERVABLE_KEYS},
        "by_topo":  {},
        "by_group": {},
    }
    by_topo_acc:  Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {k: [] for k in CORE_OBSERVABLE_KEYS})
    by_group_acc: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: {k: [] for k in CORE_OBSERVABLE_KEYS})

    n_runs_processed = 0
    for run_id, rows in per_run_rows.items():
        try:
            run_obs = extract_run_observables(rows)
        except Exception as e:
            if verbose:
                print(f"  [warn] skipping run {run_id}: {e}")
            continue
        n_runs_processed += 1
        key = run_group_key[run_id]
        for legacy_key, vals in run_obs.items():
            result["pooled"][legacy_key].extend(vals)
            by_topo_acc[key[0]][legacy_key].extend(vals)
            by_group_acc[key][legacy_key].extend(vals)

    result["pooled"]["influence_per_agent"] = extract_influence_per_agent(all_events_flat)

    if verbose:
        for obs_name, sizes in result["pooled"].items():
            xmax = float(max(sizes)) if sizes else 0.0
            print(f"  pooled/{obs_name:<24} n={len(sizes):6d}  xmax={xmax:.1f}")

    for topo, obs_dict in by_topo_acc.items():
        result["by_topo"][topo] = dict(obs_dict)
        topo_flat = [ev for ev in all_events_flat if ev.get("topology") == topo]
        result["by_topo"][topo]["influence_per_agent"] = extract_influence_per_agent(topo_flat)

    for (topo, bench, n), obs_dict in by_group_acc.items():
        key_str = f"{topo}__{bench}__n{n}"
        result["by_group"][key_str] = dict(obs_dict)
        group_flat = [
            ev for ev in all_events_flat
            if ev.get("topology") == topo and ev.get("benchmark") == bench
            and ev.get("num_agents") == n
        ]
        result["by_group"][key_str]["influence_per_agent"] = extract_influence_per_agent(group_flat)

    print(f"  Processed {n_runs_processed} runs across {len(result['by_group'])} "
          f"(topology, benchmark, N) groups")
    return result
