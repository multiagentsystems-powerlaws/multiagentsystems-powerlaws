#!/usr/bin/env python3
"""
scripts/generate_figures.py
-----------------------------
Generate all paper figures from sweep results.

Figures are saved as PDF + PNG to --out directory.
Run after extract_events.py and run_analysis.py.

Usage
-----
  # All figures from full sweep:
  python scripts/generate_figures.py \\
      --data-roots /scratch/.../MARBLE /scratch/.../GAIA \\
                   /scratch/.../REALM  /scratch/.../SWE-bench \\
      --out figures/

  # Subset of figures:
  python scripts/generate_figures.py \\
      --data-roots data/runs --out figures/ \\
      --figs ccdf_panel topology_comparison lorenz_curves

  # Single run DAG / raster for a specific run directory:
  python scripts/generate_figures.py \\
      --data-roots data/runs --out figures/ \\
      --run-dir data/runs/chain/n8/s0/task_001 \\
      --figs reasoning_dag temporal_burst

Figure catalogue
----------------
H1 — Power-law structure:
  ccdf_panel              Multi-panel CCDF by topology (anchor figure)
  powerlaw_fit            PL vs lognormal comparison per benchmark
  lr_heatmap              LR test heatmap (observable × alternative)
  ccdf_by_family          CCDF per task family (universality check)
  topology_comparison     Fitted alpha ± sigma per topology

H2 — Coordination quality:
  regime_phase            Phase space map (Gini × coalition persistence)
  lorenz_curves           Lorenz curves of agent influence
  event_vs_success        Event size vs task success
  cognitive_work          W ~ S^beta scaling
  coalition_sankey        Benchmark → topology → event → outcome flow
  coalition_timeline      Coalition evolution over time (one run)
  error_correction        Error-correction avalanche case study

H3 — Scaling:
  avalanche_size_duration S ~ T^gamma dynamical scaling
  avalanche_heatmap       Agent × step burst density
  temporal_burst          Raster plot for best run
  entropy_vs_alpha        Graph entropy vs alpha
  path_length_vs_alpha    APL vs alpha
  mega_cascade            Largest cascade tree
  reasoning_dag           Claim DAG for one run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALL_FIGS = [
    # H1
    "ccdf_panel", "powerlaw_fit", "lr_heatmap", "ccdf_by_family",
    "topology_comparison",
    # H2
    "regime_phase", "lorenz_curves", "event_vs_success", "cognitive_work",
    "coalition_sankey", "coalition_timeline", "error_correction",
    # H3
    "avalanche_size_duration", "avalanche_heatmap", "temporal_burst",
    "entropy_vs_alpha", "path_length_vs_alpha", "mega_cascade", "reasoning_dag",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-roots", nargs="+", required=True, metavar="PATH")
    p.add_argument("--out", default="figures", metavar="PATH")
    p.add_argument("--figs", nargs="+", default=None, metavar="FIG",
                   choices=ALL_FIGS,
                   help="Subset of figures to generate (default: all)")
    p.add_argument("--run-dir", default=None, metavar="PATH",
                   help="Specific run directory for per-run figures "
                        "(reasoning_dag, temporal_burst)")
    p.add_argument("--topology", default="dynamic_reputation",
                   help="Topology for timeline/heatmap figures (default: dynamic_reputation)")
    p.add_argument("--observable", default="delegation_cascade",
                   help="Observable for CCDF figures (default: delegation_cascade)")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    roots     = [Path(r) for r in args.data_roots]
    out       = Path(args.out)
    figs      = args.figs or ALL_FIGS
    run_dir   = Path(args.run_dir) if args.run_dir else None

    out.mkdir(parents=True, exist_ok=True)

    # Validate roots
    missing = [r for r in roots if not r.exists()]
    if missing:
        print(f"WARNING: these data roots don't exist yet: {missing}")

    print(f"\nGenerating {len(figs)} figures → {out}/")
    print(f"  data-roots:  {[r.name for r in roots]}")
    print(f"  topology:    {args.topology}")
    print(f"  observable:  {args.observable}")
    if run_dir:
        print(f"  run-dir:     {run_dir}")
    print()

    errors = []

    def _run(fig_id: str, fn, **kwargs):
        try:
            fn(**kwargs)
        except Exception as e:
            import traceback
            print(f"  ERROR in {fig_id}: {e}")
            traceback.print_exc()
            errors.append(fig_id)

    # ── H1 figures ────────────────────────────────────────────────

    if "ccdf_panel" in figs:
        from visualization.ccdf_panel import plot_ccdf_panel
        _run("ccdf_panel", plot_ccdf_panel,
             data_roots=roots, out_dir=out,
             observable=args.observable)

    if "powerlaw_fit" in figs:
        from visualization.powerlaw_fit_plot import plot_fit_comparison
        _run("powerlaw_fit", plot_fit_comparison,
             data_roots=roots, out_dir=out,
             observable=args.observable)

    if "lr_heatmap" in figs:
        from visualization.lr_test_heatmap import plot_lr_heatmap
        _run("lr_heatmap", plot_lr_heatmap,
             data_roots=roots, out_dir=out)

    if "ccdf_by_family" in figs:
        from visualization.ccdf_by_family import plot_ccdf_by_family
        _run("ccdf_by_family", plot_ccdf_by_family,
             data_roots=roots, out_dir=out,
             observable=args.observable)

    if "topology_comparison" in figs:
        from visualization.topology_comparison import plot_topology_comparison
        _run("topology_comparison", plot_topology_comparison,
             data_roots=roots, out_dir=out,
             observable=args.observable)

    # ── H2 figures ────────────────────────────────────────────────

    if "regime_phase" in figs:
        from visualization.regime_phase_diagram import plot_regime_phase_diagram
        _run("regime_phase", plot_regime_phase_diagram,
             data_roots=roots, out_dir=out)

    if "lorenz_curves" in figs:
        from visualization.lorenz_curves import plot_lorenz_curves
        _run("lorenz_curves", plot_lorenz_curves,
             data_roots=roots, out_dir=out)

    if "event_vs_success" in figs:
        from visualization.event_vs_success import plot_event_vs_success
        _run("event_vs_success", plot_event_vs_success,
             data_roots=roots, out_dir=out)

    if "cognitive_work" in figs:
        from visualization.cognitive_work_scaling import plot_cognitive_work_scaling
        _run("cognitive_work", plot_cognitive_work_scaling,
             data_roots=roots, out_dir=out)

    if "coalition_sankey" in figs:
        from visualization.coalition_sankey import plot_coalition_sankey
        _run("coalition_sankey", plot_coalition_sankey,
             data_roots=roots, out_dir=out)

    if "coalition_timeline" in figs:
        from visualization.coalition_timeline import plot_coalition_timeline
        _run("coalition_timeline", plot_coalition_timeline,
             data_roots=roots, out_dir=out,
             topology=args.topology)

    if "error_correction" in figs:
        from visualization.error_correction_avalanche import plot_error_correction_avalanche
        _run("error_correction", plot_error_correction_avalanche,
             data_roots=roots, out_dir=out)

    # ── H3 figures ────────────────────────────────────────────────

    if "avalanche_size_duration" in figs:
        from visualization.avalanche_size_duration import plot_avalanche_size_duration
        _run("avalanche_size_duration", plot_avalanche_size_duration,
             data_roots=roots, out_dir=out)

    if "avalanche_heatmap" in figs:
        from visualization.avalanche_raster import plot_avalanche_heatmap
        _run("avalanche_heatmap", plot_avalanche_heatmap,
             data_roots=roots, out_dir=out,
             topology=args.topology)

    if "temporal_burst" in figs:
        from visualization.temporal_burst_plot import (
            plot_avalanche_raster, plot_raster_for_best_run
        )
        if run_dir and run_dir.exists():
            _run("temporal_burst", plot_avalanche_raster,
                 run_dir=run_dir, out_dir=out)
        else:
            _run("temporal_burst", plot_raster_for_best_run,
                 data_roots=roots, out_dir=out,
                 topology=args.topology)

    if "entropy_vs_alpha" in figs:
        from visualization.entropy_vs_alpha import plot_entropy_vs_alpha
        _run("entropy_vs_alpha", plot_entropy_vs_alpha,
             data_roots=roots, out_dir=out,
             observable=args.observable)

    if "path_length_vs_alpha" in figs:
        from visualization.path_length_vs_alpha import plot_path_length_vs_alpha
        _run("path_length_vs_alpha", plot_path_length_vs_alpha,
             data_roots=roots, out_dir=out,
             observable=args.observable)

    if "mega_cascade" in figs:
        from visualization.mega_cascade import plot_mega_cascade
        _run("mega_cascade", plot_mega_cascade,
             data_roots=roots, out_dir=out)

    if "reasoning_dag" in figs:
        from visualization.reasoning_dag_viz import plot_reasoning_dag
        if run_dir and run_dir.exists():
            _run("reasoning_dag", plot_reasoning_dag,
                 run_dir=run_dir, out_dir=out)
        else:
            # Find the run with the most events automatically
            best_path, best_count = None, 0
            for root in roots:
                for ep in root.rglob("events.jsonl"):
                    try:
                        count = sum(1 for _ in open(ep))
                        if count > best_count:
                            best_count = count
                            best_path  = ep.parent
                    except Exception:
                        pass
            if best_path:
                _run("reasoning_dag", plot_reasoning_dag,
                     run_dir=best_path, out_dir=out)
            else:
                print("  reasoning_dag: no runs found — use --run-dir")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    n_ok = len(figs) - len(errors)
    print(f"Done: {n_ok}/{len(figs)} figures saved to {out}/")
    if errors:
        print(f"Failed: {errors}")
        print("  (check that data-roots exist and sweep has completed)")


if __name__ == "__main__":
    main()