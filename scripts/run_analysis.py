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
        # Run extraction inline via the shared, correctness-verified pipeline
        # (event_extractor -> resolve -> dag_builder -> cascade_metrics) --
        # same implementation used by extract_events.py and main.py's
        # stage_extract, not the coordination.py path (which gets
        # delegation/revision/merge/TCE wrong — see pooled_observables.py).
        print("No pre-extracted observables found; running extraction...")
        from analysis.pooled_observables import pool_h1_observables

        roots = [Path(r) for r in args.data_roots]
        h1 = pool_h1_observables(
            roots, topology_filter=args.topology, benchmark_filter=args.benchmark,
            n_filter=args.n, verbose=args.verbose,
        )
        return h1["pooled"]

    print("ERROR: provide --observables or --data-roots")
    sys.exit(1)


def _fit_result_to_dict(r: FitResult) -> Dict:
    """Serialize FitResult to a JSON-safe dict."""
    return {
        "observable":               r.observable,
        "n_total":                  r.n_total,
        "n_tail":                   r.n_tail,
        "tail_fraction":            r.tail_fraction,
        "x_min":                    r.x_min,
        "alpha":                    r.alpha,
        "alpha_ci_low":             r.alpha_ci_low,
        "alpha_ci_high":            r.alpha_ci_high,
        "ci_method":                r.ci_method,
        "x_c":                      r.x_c,
        # KS distance — NOT a p-value
        "ks_stat":                  r.ks_stat,
        # Semiparametric GOF for the fitted TPL
        "gof_pvalue":               r.gof_pvalue,
        "gof_method":               r.gof_method,
        # Non-nested Vuong tests (TPL vs LN / exponential)
        "lr_tpl_vs_ln":             r.lr_tpl_vs_ln,
        "lr_tpl_vs_ln_p":           r.lr_tpl_vs_ln_p,
        "lr_tpl_vs_exp":            r.lr_tpl_vs_exp,
        "lr_tpl_vs_exp_p":          r.lr_tpl_vs_exp_p,
        # Boundary-aware nested LRT (TPL vs PL)
        "lr_tpl_vs_pl":             r.lr_tpl_vs_pl,
        "lr_tpl_vs_pl_p":           r.lr_tpl_vs_pl_p,
        # Interpretability
        "tpl_plausible":            r.tpl_plausible,
        "tpl_beats_lognormal":      r.tpl_beats_lognormal,
        "tpl_beats_pl":             r.tpl_beats_pl,
        "regime":                   r.regime,
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
    n_tpl      = sum(1 for r in fits.values() if r.tpl_plausible)
    n_beats_pl = sum(1 for r in fits.values() if r.tpl_beats_pl)
    print(f"\n{'='*60}")
    print(f"Analysis complete → {out}/")
    print(f"  Fitted: {len(fits)}/{len(obs)} observables")
    print(f"  TPL plausible (GOF):          {n_tpl}")
    print(f"  TPL beats PL (truncation real): {n_beats_pl}")
    print(f"\nNext: python scripts/generate_tables.py --data-roots ... --analysis {out}")


if __name__ == "__main__":
    main()