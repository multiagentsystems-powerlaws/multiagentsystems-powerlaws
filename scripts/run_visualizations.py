#!/usr/bin/env python3
"""
scripts/run_visualizations.py
------------------------------
Generate all 16 paper figures from sweep results.

Usage
-----
  # All figures from all benchmark roots
  python scripts/run_visualizations.py \\
      --data-roots \\
        /scratch/results/MARBLE \\
        /scratch/results/GAIA \\
        /scratch/results/REALM \\
        /scratch/results/SWE-bench \\
      --out figures/

  # Single root (local dev / smoke test)
  python scripts/run_visualizations.py \\
      --data-roots data/runs \\
      --out figures/

  # Specific figures only
  python scripts/run_visualizations.py \\
      --data-roots data/runs --out figures/ \\
      --figs ccdf_panel xmax_scaling lorenz_curves

  # Skip figures that need >= N samples
  python scripts/run_visualizations.py \\
      --data-roots data/runs --out figures/ \\
      --min-samples 30
"""

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-roots", nargs="+", required=True,
                   metavar="PATH", help="One or more benchmark data-root directories")
    p.add_argument("--out", default="figures", metavar="PATH",
                   help="Output directory for PDF/PNG figures")
    p.add_argument("--figs", nargs="+", default=None,
                   metavar="NAME",
                   help="Run only these figures (default: all)")
    p.add_argument("--topology", default=None,
                   help="Filter to one topology for raster/heatmap figures")
    p.add_argument("--observable", default="delegation_cascade",
                   help="Observable to use for CCDF/universality figures")
    p.add_argument("--min-samples", type=int, default=20,
                   help="Minimum samples needed to attempt a fit")
    p.add_argument("--dry-run", action="store_true",
                   help="List figures that would be generated without running them")
    return p.parse_args()


def bold(s):  return f"\033[1m{s}\033[0m"
def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"


def main():
    args     = parse_args()
    roots    = [Path(r) for r in args.data_roots]
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify roots
    missing = [r for r in roots if not r.exists()]
    if missing:
        print(yellow(f"Warning: {len(missing)} root(s) not found:"))
        for r in missing:
            print(f"  {r}")
    roots = [r for r in roots if r.exists()]
    if not roots:
        print(red("Error: no valid data roots found."))
        sys.exit(1)

    # Count available events
    total_events = sum(len(list(r.rglob("events.jsonl"))) for r in roots)
    print(bold(f"\n╔══════════════════════════════════════╗"))
    print(bold(f"║   MAS Power-Law — Figure Generator   ║"))
    print(bold(f"╚══════════════════════════════════════╝"))
    print(f"  Data roots:    {[r.name for r in roots]}")
    print(f"  Events files:  {total_events}")
    print(f"  Output dir:    {out_dir}")
    print(f"  Observable:    {args.observable}")

    # ── Figure registry ───────────────────────────────────────────────
    # Each entry: (short_name, description, callable)
    # Callables are defined below using lazy imports so failures in one
    # module don't block others.

    def fig_ccdf_panel():
        from visualization.ccdf_panel import plot_ccdf_panel
        return plot_ccdf_panel(
            data_roots=roots, out_dir=out_dir,
            observable=args.observable,
            min_samples=args.min_samples,
        )

    def fig_fit_comparison():
        from visualization.powerlaw_fit_plot import plot_fit_comparison
        return plot_fit_comparison(
            data_roots=roots, out_dir=out_dir,
            observable=args.observable,
        )

    def fig_lr_heatmap():
        from visualization.lr_test_heatmap import plot_lr_heatmap
        return plot_lr_heatmap(data_roots=roots, out_dir=out_dir)

    def fig_universality():
        from visualization.universality_collapse import plot_universality_collapse
        return plot_universality_collapse(
            data_roots=roots, out_dir=out_dir,
            observable=args.observable,
        )

    def fig_xmax_scaling():
        from visualization.xmax_scaling import plot_xmax_scaling
        return plot_xmax_scaling(
            data_roots=roots, out_dir=out_dir,
            observable=args.observable,
        )

    def fig_entropy_alpha():
        from visualization.entropy_vs_alpha import plot_entropy_vs_alpha
        return plot_entropy_vs_alpha(
            data_roots=roots, out_dir=out_dir,
            observable=args.observable,
        )

    def fig_temporal_burst():
        from visualization.temporal_burst_plot import plot_raster_for_best_run
        topo = args.topology or "dynamic_reputation"
        return plot_raster_for_best_run(
            data_roots=roots, out_dir=out_dir, topology=topo,
        )

    def fig_reasoning_dag():
        from visualization.reasoning_dag_viz import plot_reasoning_dag
        # Find the run with the most events
        best_path  = None
        best_count = 0
        for root in roots:
            for ep in root.rglob("events.jsonl"):
                count = sum(1 for _ in open(ep))
                if count > best_count:
                    best_count = count
                    best_path  = ep.parent
        if best_path is None:
            print("  reasoning_dag: no runs found")
            return None
        return plot_reasoning_dag(run_dir=best_path, out_dir=out_dir)

    def fig_phase_diagram():
        from visualization.regime_phase_diagram import plot_regime_phase_diagram
        return plot_regime_phase_diagram(data_roots=roots, out_dir=out_dir)

    def fig_sankey():
        from visualization.coalition_sankey import plot_coalition_sankey
        return plot_coalition_sankey(data_roots=roots, out_dir=out_dir)

    def fig_lorenz():
        from visualization.lorenz_curves import plot_lorenz_curves
        return plot_lorenz_curves(data_roots=roots, out_dir=out_dir)

    def fig_topo_comparison():
        from visualization.topology_comparison import plot_topology_comparison
        return plot_topology_comparison(
            data_roots=roots, out_dir=out_dir,
            observable=args.observable,
        )

    def fig_avalanche_raster():
        from visualization.temporal_burst_plot import plot_raster_for_best_run
        topo = args.topology or "dynamic_reputation"
        return plot_raster_for_best_run(
            data_roots=roots, out_dir=out_dir,
            topology=topo, figname="fig_avalanche_raster_scatter",
        )

    def fig_avalanche_heatmap():
        from visualization.avalanche_raster import plot_avalanche_heatmap
        topo = args.topology or "dynamic_reputation"
        return plot_avalanche_heatmap(
            data_roots=roots, out_dir=out_dir, topology=topo,
        )

    def fig_work_scaling():
        from visualization.cognitive_work_scaling import plot_cognitive_work_scaling
        return plot_cognitive_work_scaling(data_roots=roots, out_dir=out_dir)

    def fig_event_success():
        from visualization.event_vs_success import plot_event_vs_success
        return plot_event_vs_success(data_roots=roots, out_dir=out_dir)

    def fig_mega_cascade():
        from visualization.mega_cascade import plot_mega_cascade
        return plot_mega_cascade(data_roots=roots, out_dir=out_dir)


    def fig_ccdf_by_family():
        from visualization.ccdf_by_family import plot_ccdf_by_family
        return plot_ccdf_by_family(data_roots=roots, out_dir=out_dir,
                                   observable=args.observable,
                                   min_samples=args.min_samples)

    def fig_mean_vs_max():
        from visualization.mean_vs_max_scaling import plot_mean_vs_max_scaling
        return plot_mean_vs_max_scaling(data_roots=roots, out_dir=out_dir,
                                        observable=args.observable)

    def fig_size_duration():
        from visualization.avalanche_size_duration import plot_avalanche_size_duration
        return plot_avalanche_size_duration(data_roots=roots, out_dir=out_dir)

    def fig_coalition_timeline():
        from visualization.coalition_timeline import plot_coalition_timeline
        topo = args.topology or "dynamic_reputation"
        return plot_coalition_timeline(data_roots=roots, out_dir=out_dir, topology=topo)

    def fig_path_length_alpha():
        from visualization.path_length_vs_alpha import plot_path_length_vs_alpha
        return plot_path_length_vs_alpha(data_roots=roots, out_dir=out_dir,
                                          observable=args.observable)

    def fig_error_correction():
        from visualization.error_correction_avalanche import plot_error_correction_avalanche
        return plot_error_correction_avalanche(data_roots=roots, out_dir=out_dir)

    def fig_appendix_all():
        from visualization.appendix.appendix_figures import plot_all_appendix_figures
        plot_all_appendix_figures(roots, out_dir / "appendix")
        return True

    # ── Ordered figure list ───────────────────────────────────────────
    ALL_FIGS = [
        ("ccdf_panel",        "Fig 2: Multi-panel CCDF (anchor figure)",         fig_ccdf_panel),
        ("fit_comparison",    "Fig 2b: PL vs lognormal overlay",                 fig_fit_comparison),
        ("lr_heatmap",        "Fig 4: Likelihood-ratio heatmap",                 fig_lr_heatmap),
        ("universality",      "Fig 3: Universality collapse",                    fig_universality),
        ("xmax_scaling",      "Fig 5: Extreme event scaling x_max vs N",         fig_xmax_scaling),
        ("entropy_alpha",     "Fig: Graph entropy vs alpha",                     fig_entropy_alpha),
        ("temporal_burst",    "Fig 6: Cognitive avalanche raster (scatter)",     fig_temporal_burst),
        ("reasoning_dag",     "Fig: Reasoning DAG",                             fig_reasoning_dag),
        ("phase_diagram",     "Fig 8: MAS phase space map",                      fig_phase_diagram),
        ("sankey",            "Fig: Coalition Sankey flow",                      fig_sankey),
        ("lorenz",            "Fig: Lorenz curves of influence",                 fig_lorenz),
        ("topo_comparison",   "Fig: Topology alpha comparison",                  fig_topo_comparison),
        ("avalanche_raster",  "Fig: Avalanche raster scatter",                   fig_avalanche_raster),
        ("avalanche_heatmap", "Fig: Avalanche heatmap",                          fig_avalanche_heatmap),
        ("work_scaling",      "Fig: Cognitive work scaling W~S^beta",            fig_work_scaling),
        ("event_success",     "Fig: Event size vs task success",                 fig_event_success),
        ("mega_cascade",      "Fig: Mega cascade case study",                    fig_mega_cascade),
        ("ccdf_by_family",    "Fig: CCDF by task family (V2)",                  fig_ccdf_by_family),
        ("mean_vs_max",       "Fig: Mean vs max scaling (V6)",                  fig_mean_vs_max),
        ("size_duration",     "Fig: Avalanche size vs duration S~T^γ (V8)",     fig_size_duration),
        ("coalition_timeline","Fig: Coalition evolution timeline (V10)",         fig_coalition_timeline),
        ("path_length_alpha", "Fig: Path length vs tail exponent (V12)",         fig_path_length_alpha),
        ("error_correction",  "Fig: Error-correction avalanche (V18)",            fig_error_correction),
        ("appendix_all",      "Appendix A1-A6: all robustness figures",           fig_appendix_all),
    ]

    # Filter by --figs if specified
    if args.figs:
        fig_names = set(args.figs)
        run_figs  = [(n, d, fn) for n, d, fn in ALL_FIGS if n in fig_names]
        missing_f = fig_names - {n for n, _, _ in ALL_FIGS}
        if missing_f:
            print(yellow(f"Unknown figure names: {missing_f}"))
            print(f"Valid names: {[n for n,_,_ in ALL_FIGS]}")
    else:
        run_figs = ALL_FIGS

    if args.dry_run:
        print(f"\nWould generate {len(run_figs)} figures:")
        for name, desc, _ in run_figs:
            print(f"  {name:<22}  {desc}")
        print(f"\nOutput: {out_dir}/{{name}}.pdf  +  .png")
        return

    # ── Run all figures ───────────────────────────────────────────────
    print(f"\nGenerating {len(run_figs)} figures...\n")
    passed, failed, skipped = [], [], []

    for name, desc, fn in run_figs:
        print(f"  {desc}")
        try:
            result = fn()
            if result is None:
                print(yellow(f"    ○ SKIP — function returned None (not enough data)"))
                skipped.append(name)
            else:
                print(green(f"    ✓ DONE"))
                passed.append(name)
        except Exception:
            tb = traceback.format_exc()
            print(red(f"    ✗ FAIL"))
            print(f"      {tb.splitlines()[-1]}")
            failed.append((name, tb))

    # ── Summary ───────────────────────────────────────────────────────
    print(bold(f"\n── Summary {'─'*35}"))
    print(f"  Generated:  {len(passed)}")
    print(f"  Skipped:    {len(skipped)}  (insufficient data — will fill as sweep grows)")
    print(f"  Failed:     {len(failed)}")

    if failed:
        print(red(f"\n  Failed figures:"))
        for name, tb in failed:
            print(f"    {name}:")
            for line in tb.splitlines()[-4:]:
                print(f"      {line}")

    print(f"\n  Output: {out_dir.resolve()}/")
    pdfs = list(out_dir.glob("*.pdf"))
    print(f"  PDF files: {len(pdfs)}")


if __name__ == "__main__":
    main()
