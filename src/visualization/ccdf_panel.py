"""
src/visualization/ccdf_panel.py

CCDF panels for all five coordination observables.

Produces:
  - One panel per observable
  - Empirical CCDF as scatter, one color per topology/condition
  - MLE power-law fit overlay above x_min
  - Log-normal fit overlay for comparison
  - alpha annotation on fit line
  - x_min indicated by vertical dashed line

Input: pooled event_observables dict + FitResult dict
       (from analysis/run_pipeline.process_condition)

Usage:
    from analysis.run_pipeline import process_condition, discover_run_dirs
    from tail_fitting.powerlaw_fit import fit_all
    from visualization.ccdf_panel import plot_ccdf_panel

    fits = process_condition(run_dirs, "gaia_chain_n64")
    pooled_obs = pool_observables(run_dirs)  # see run_pipeline
    plot_ccdf_panel(pooled_obs, fits, out_dir=Path("figures/"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

from tail_fitting.powerlaw_fit import (
    FitResult,
    empirical_ccdf,
    powerlaw_ccdf_line,
    OBSERVABLES,
)

try:
    import powerlaw as pl_pkg  # type: ignore
    _HAS_POWERLAW = True
except ImportError:
    _HAS_POWERLAW = False


# ── Style constants ───────────────────────────────────────────

TOPOLOGY_COLORS = {
    "chain":            "#4878CF",
    "star":             "#D65F5F",
    "tree":             "#6ACC65",
    "full_mesh":        "#B47CC7",
    "sparse_mesh":      "#C4AD66",
    "hybrid_modular":   "#77BEDB",
    "dynamic_reputation":"#F7A551",
}

OBSERVABLE_ORDER = [
    "delegation_sizes",
    "revision_waves",
    "contradiction_bursts",
    "merge_fan_in",
    "tce",
]


def _apply_paper_style() -> None:
    """Minimal rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   7,
        "figure.dpi":        150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "lines.linewidth":   1.2,
    })


# ── Main figure function ──────────────────────────────────────

def plot_ccdf_panel(
    pooled_observables: Dict[str, List[int]],
    fits:               Dict[str, FitResult],
    out_dir:            Path,
    condition_label:    str = "",
    topology_label:     str = "",
    color:              str = "#4878CF",
    figname:            str = "ccdf_panel",
    show_lognormal:     bool = True,
    figsize:            tuple = (12, 3),
) -> plt.Figure:
    """
    Plot CCDF panel for all five observables.

    Args:
        pooled_observables: event_observables dict from cascade_metrics.
        fits:               FitResult dict from powerlaw_fit.fit_all().
        out_dir:            directory to save figure.
        condition_label:    appended to suptitle.
        topology_label:     used for legend entry.
        color:              line/scatter color for this condition.
        figname:            output filename stem.
        show_lognormal:     overlay log-normal fit for comparison.
        figsize:            figure size in inches.

    Returns the matplotlib Figure.
    """
    _apply_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_obs = len(OBSERVABLE_ORDER)
    fig, axes = plt.subplots(1, n_obs, figsize=figsize)

    for ax, obs_name in zip(axes, OBSERVABLE_ORDER):
        data = [x for x in pooled_observables.get(obs_name, []) if x > 0]
        fit  = fits.get(obs_name)
        meta = OBSERVABLES.get(obs_name, {"label": obs_name})

        if not data:
            ax.set_visible(False)
            continue

        # Empirical CCDF
        x_emp, p_emp = empirical_ccdf(data)
        p_emp = np.maximum.accumulate(p_emp[::-1])[::-1]
        label = topology_label or condition_label or "data"
        ax.loglog(x_emp, p_emp, ".", color=color, alpha=0.5, ms=2,
                  rasterized=True, label=label)

        if fit is not None:
            # x_min indicator
            ax.axvline(fit.x_min, color=color, lw=0.8, ls="--", alpha=0.6)

            # Power-law fit overlay (normalised to empirical at x_min)
            x_fit, ccdf_fit = powerlaw_ccdf_line(fit.x_min, float(max(x_emp)),
                                                  fit.alpha)
            # Find empirical CCDF value at x_min for normalisation
            xm_idx = np.searchsorted(x_emp, fit.x_min, side="left")
            xm_idx = min(xm_idx, len(p_emp) - 1)
            scale = p_emp[xm_idx]
            if xm_idx < len(p_emp) and p_emp[xm_idx] > 0:
                scale = p_emp[xm_idx]
                ax.loglog(x_fit, ccdf_fit * scale, "-", color=color, lw=1.5,
                          label=f"PL α={fit.alpha:.2f}")

            # Log-normal overlay for comparison
            if show_lognormal and _HAS_POWERLAW:
                try:
                    arr = np.array(data, dtype=float)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        plfit = pl_pkg.Fit(arr, xmin=fit.x_min, discrete=True,
                                           verbose=False)
                    tail = arr[arr >= fit.x_min]
                    x_ln = np.logspace(np.log10(fit.x_min), np.log10(arr.max()), 200)
                    p_ln = np.array([plfit.lognormal.ccdf(xi) for xi in x_ln])
                    # Normalise at x_min
                    if p_ln[0] > 0 and scale > 0:
                        p_ln = p_ln * (scale / p_ln[0])
                    ax.loglog(x_ln, p_ln, "--", color=color, lw=1.0, alpha=0.5,
                              label="LogNorm")
                except Exception:
                    pass

            # Annotation: alpha ± sigma
            ax.text(0.97, 0.95,
                    f"α={fit.alpha:.2f}±{fit.sigma_alpha:.2f}\n"
                    f"n={fit.n_tail}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=7,
                    color=color)

        ax.set_title(meta["label"], fontsize=8, pad=4)
        ax.set_xlabel("Event size $x$", fontsize=8)
        ax.set_ylabel("$P(X \\geq x)$", fontsize=8)
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        ax.tick_params(which="both", labelsize=7)

    suptitle = "CCDF of coordination event sizes"
    if condition_label:
        suptitle += f" — {condition_label}"
    fig.suptitle(suptitle, fontsize=9, y=1.02)
    fig.tight_layout()

    out_path = out_dir / f"{figname}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {out_path}")
    return fig


# ── Multi-topology overlay ────────────────────────────────────

def plot_ccdf_by_topology(
    topology_observables: Dict[str, Dict[str, List[int]]],
    topology_fits:        Dict[str, Dict[str, FitResult]],
    out_dir:              Path,
    observable:           str = "tce",
    figname:              str = "ccdf_by_topology",
    figsize:              tuple = (5, 4),
) -> plt.Figure:
    """
    Single observable, all topologies overlaid on one axes.
    topology_observables: {topology_name: event_observables_dict}
    topology_fits:        {topology_name: fits_dict}
    """
    _apply_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    meta = OBSERVABLES.get(observable, {"label": observable})

    for topo, color in TOPOLOGY_COLORS.items():
        data = [x for x in topology_observables.get(topo, {}).get(observable, [])
                if x > 0]
        if not data:
            continue

        x_emp, p_emp = empirical_ccdf(data)
        ax.loglog(x_emp, p_emp, ".", color=color, alpha=0.4, ms=2, rasterized=True)

        fit = topology_fits.get(topo, {}).get(observable)
        if fit is not None:
            x_fit, ccdf_fit = powerlaw_ccdf_line(fit.x_min, float(max(x_emp)),
                                                  fit.alpha)
            xm_idx = np.searchsorted(x_emp, fit.x_min, side="left")
            xm_idx = min(xm_idx, len(p_emp) - 1)
            scale = p_emp[xm_idx]
            if xm_idx < len(p_emp) and p_emp[xm_idx] > 0:
                scale = p_emp[xm_idx]
                ax.loglog(x_fit, ccdf_fit * scale, "-", color=color, lw=1.5,
                          label=f"{topo} α={fit.alpha:.2f}")
            ax.axvline(fit.x_min, color=color, lw=0.6, ls=":", alpha=0.5)

    ax.set_title(f"CCDF: {meta['label']}", fontsize=9)
    ax.set_xlabel("Event size $x$", fontsize=9)
    ax.set_ylabel("$P(X \\geq x)$", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(which="both", labelsize=7)
    fig.tight_layout()

    out_path = out_dir / f"{figname}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {out_path}")
    return fig