#!/usr/bin/env python3
"""
scripts/run_analysis.py
------------------------
After all GPU sweep jobs finish, load extracted observables and run
power-law fitting. Produces fit_results.json and table1_fits.csv.

Workflow:
  1. Run extract_events.py first  → data/extracted/
  2. Run this script              → data/analysis/

Usage
-----
  # Full pipeline:
  python scripts/extract_events.py \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA \\
                   /scratch/.../REALM  /scratch/.../SWE-bench \\
      --out data/extracted

  python scripts/run_analysis.py \\
      --observables data/extracted/observables_pooled.json \\
      --by-group    data/extracted/observables_by_group.json \\
      --out         data/analysis

  # Or point directly at data-roots (will call extraction internally):
  python scripts/run_analysis.py \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA \\
      --out data/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tail_fitting.powerlaw_fit import fit_all, fits_to_dataframe, FitResult, OBS_SPECS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Power-law analysis after sweep")
    # Option A: point at pre-extracted JSON
    p.add_argument("--observables", default=None, metavar="PATH",
                   help="Path to observables_pooled.json from extract_events.py")
    p.add_argument("--by-group",    default=None, metavar="PATH",
                   help="Path to observables_by_group.json from extract_events.py")
    # Option B: point at raw data roots (extraction runs inline)
    p.add_argument("--data-roots",  nargs="+", default=None, metavar="PATH")
    p.add_argument("--topology",    default=None)
    p.add_argument("--benchmark",   default=None)
    p.add_argument("--n",           type=int, default=None)
    # Output
    p.add_argument("--out",         default="data/analysis", metavar="PATH")
    # Fitting thresholds
    p.add_argument("--min-n",       type=int, default=50,
                   help="Min samples for fitting (lower for exploratory runs)")
    p.add_argument("--min-tail",    type=int, default=50)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def _load_observables(args) -> Dict[str, List[float]]:
    """Load pooled observables — from JSON or by running extraction inline."""
    if args.observables and Path(args.observables).exists():
        print(f"Loading observables from {args.observables}")
        obs = json.loads(Path(args.observables).read_text())
        print(f"  {len(obs)} observables loaded")
        return obs

    if args.data_roots:
        # Run extraction inline
        print("No pre-extracted observables found; running extraction...")
        from event_extraction.coordination import (
            _load_all_events, filter_events,
            extract_delegation_cascades, extract_revision_waves,
            extract_contradiction_bursts, extract_merge_fanin,
            extract_influence_per_agent,
        )
        from event_extraction.tce import tce_per_run
        from collections import defaultdict

        all_obs: Dict[str, List[float]] = defaultdict(list)
        for root_str in args.data_roots:
            root = Path(root_str)
            if not root.exists():
                continue
            evs = filter_events(
                _load_all_events(root),
                topology=args.topology,
                benchmark=args.benchmark,
                num_agents=args.n,
            )
            for name, fn in {
                "delegation_cascade":  extract_delegation_cascades,
                "revision_wave":       extract_revision_waves,
                "contradiction_burst": extract_contradiction_bursts,
                "merge_fanin":         extract_merge_fanin,
                "influence_per_agent": extract_influence_per_agent,
            }.items():
                all_obs[name].extend(fn(evs))
            all_obs["tce_per_run"].extend(tce_per_run(root))
        return dict(all_obs)

    print("ERROR: provide --observables or --data-roots")
    sys.exit(1)


def _fit_result_to_dict(r: FitResult) -> Dict:
    """Serialize FitResult to JSON-safe dict using new field names."""
    return {
        "observable":               r.observable,
        "n_total":                  r.n_total,
        "n_tail":                   r.n_tail,
        "tail_fraction":            r.tail_fraction,
        "x_min":                    r.x_min,
        "alpha":                    r.alpha,
        "sigma_alpha":              r.sigma_alpha,
        # KS distance — NOT a p-value
        "ks_statistic":             r.ks_statistic,
        # LR tests
        "lr_vs_lognormal":          r.lr_vs_lognormal,
        "lr_vs_lognormal_p":        r.lr_vs_lognormal_p,
        "lr_vs_exponential":        r.lr_vs_exponential,
        "lr_vs_exponential_p":      r.lr_vs_exponential_p,
        "lr_vs_truncated_pl":       r.lr_vs_truncated_pl,
        "lr_vs_truncated_pl_p":     r.lr_vs_truncated_pl_p,
        # Comparison labels
        "comparison_vs_lognormal":    r.comparison_vs_lognormal,
        "comparison_vs_exponential":  r.comparison_vs_exponential,
        "comparison_vs_truncated_pl": r.comparison_vs_truncated_pl,
        # AIC
        "aic_powerlaw":             r.aic_powerlaw,
        "aic_lognormal":            r.aic_lognormal,
        "aic_exponential":          r.aic_exponential,
        "aic_truncated_pl":         r.aic_truncated_pl,
        "best_model_by_aic":        r.best_model_by_aic,
        # OLS baseline
        "ols_loglog_slope":         r.ols_loglog_slope,
        # Interpretability
        "pl_plausible":             r.pl_plausible,
        "regime":                   r.regime,
        "quality_tag":              r.quality_tag,
        # Descriptives
        "mean":  r.mean,
        "median": r.median,
        "x_max": r.x_max,
        "gini":  r.gini,
    }


def main() -> None:
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load observables ──────────────────────────────────────────
    obs = _load_observables(args)

    print(f"\nObservable pool sizes:")
    for name, sizes in sorted(obs.items()):
        print(f"  {name:<30}  n={len(sizes):7d}")

    # ── Fit pooled observables ────────────────────────────────────
    print(f"\nFitting power laws (min_n={args.min_n}, min_tail={args.min_tail})...")
    fits = fit_all(
        obs,
        specs=OBS_SPECS,
        verbose=True,
        min_n_total=args.min_n,
        min_n_tail=args.min_tail,
    )

    # ── Save fit results ──────────────────────────────────────────
    fit_path = out / "fit_results.json"
    fit_path.write_text(json.dumps(
        {k: _fit_result_to_dict(v) for k, v in fits.items()},
        indent=2
    ))
    print(f"\nFit results → {fit_path}")

    # ── CSV (Table 1) ─────────────────────────────────────────────
    try:
        df = fits_to_dataframe(fits)
        csv_path = out / "table1_fits.csv"
        df.to_csv(csv_path, index=False)
        print(f"Table 1 CSV → {csv_path}")
    except ImportError:
        print("(pandas not available — skipping CSV)")

    # ── Per-group fits if available ───────────────────────────────
    if args.by_group and Path(args.by_group).exists():
        print("\nFitting per-group observables...")
        by_group = json.loads(Path(args.by_group).read_text())
        group_fit_rows = []
        for group_key, group_obs in by_group.items():
            parts = group_key.split("__")
            topo  = parts[0] if len(parts) > 0 else "unknown"
            bench = parts[1] if len(parts) > 1 else "unknown"
            n_str = parts[2].replace("n", "") if len(parts) > 2 else "0"
            group_fits = fit_all(
                group_obs, specs=OBS_SPECS, verbose=False,
                min_n_total=args.min_n, min_n_tail=args.min_tail,
            )
            for obs_name, r in group_fits.items():
                d = _fit_result_to_dict(r)
                d["group"] = group_key
                d["topology"] = topo
                d["benchmark"] = bench
                d["num_agents"] = int(n_str) if n_str.isdigit() else 0
                group_fit_rows.append(d)

        if group_fit_rows:
            try:
                import pandas as pd  # type: ignore
                pd.DataFrame(group_fit_rows).to_csv(
                    out / "fits_by_group.csv", index=False
                )
                print(f"Per-group fits → {out}/fits_by_group.csv ({len(group_fit_rows)} rows)")
            except ImportError:
                pass

    # ── Summary ───────────────────────────────────────────────────
    n_pl  = sum(1 for r in fits.values() if r.pl_plausible)
    n_aic = sum(1 for r in fits.values() if r.best_model_by_aic == "power_law")
    print(f"\n{'='*60}")
    print(f"Analysis complete → {out}/")
    print(f"  Fitted: {len(fits)}/{len(obs)} observables")
    print(f"  PL plausible:     {n_pl}")
    print(f"  Best by AIC (PL): {n_aic}")
    print(f"\nNext: python scripts/generate_tables.py --data-roots ... --analysis {out}")


if __name__ == "__main__":
    main()