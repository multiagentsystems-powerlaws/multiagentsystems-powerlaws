"""
visualization/powerlaw_fit_plot.py
-----------------------------------
Fitted power-law line + lognormal comparison overlay on one axes.
Used both standalone and as a sub-panel in other figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from visualization._style import apply_style, BENCH_COLORS, OBSERVABLE_LABELS
from event_extraction.coordination import (
    extract_delegation_cascades, extract_revision_waves,
    extract_contradiction_bursts, extract_merge_fanin,
    extract_influence_per_agent, _load_all_events,
)

_EXTRACTORS = {
    "delegation_cascade":  extract_delegation_cascades,
    "revision_wave":       extract_revision_waves,
    "contradiction_burst": extract_contradiction_bursts,
    "merge_fanin":         extract_merge_fanin,
    "influence_per_agent": extract_influence_per_agent,
}
from tail_fitting.powerlaw_fit import fit_observable, FitResult


def _ccdf(data: np.ndarray):
    x = np.sort(data)
    p = 1.0 - np.arange(len(x)) / len(x)
    return x, p


def plot_fit_comparison(
    data_roots:  List[Path],
    out_dir:     Path,
    observable:  str = "delegation_cascade",
    figname:     str = "fig_fit_comparison",
):
    """
    Single figure: empirical CCDF with power-law and lognormal overlays.
    One curve per benchmark, pooled across topologies.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for root in data_roots:
        if not root.exists():
            continue
        bench = root.name
        color = BENCH_COLORS.get(bench, "#666")
        events = _load_all_events(root)
        all_events = _load_all_events(root)
        extractor  = _EXTRACTORS.get(observable, extract_delegation_cascades)
        data       = np.array([s for s in extractor(all_events) if s > 0])

        if len(data) < 30:
            continue

        x, p = _ccdf(data)
        ax.loglog(x, p, ".", color=color, alpha=0.35, ms=2,
                  label=f"{bench} (n={len(data)})", rasterized=True)

        fit = fit_observable(data, bench, verbose=False)
        if fit and fit.n_tail >= 20:
            # Power-law line
            xs = np.logspace(np.log10(fit.x_min), np.log10(data.max()), 80)
            ys = (xs / fit.x_min) ** (-(fit.alpha - 1))
            xm_idx = np.searchsorted(x, fit.x_min)
            scale  = p[xm_idx] if xm_idx < len(p) else 1.0
            ax.loglog(xs, ys * scale, "-", color=color, lw=1.8,
                      label=f"PL α={fit.alpha:.2f}")

            # Lognormal fit
            try:
                import powerlaw as pl_pkg  # type: ignore
                fit_obj = pl_pkg.Fit(data, xmin=fit.x_min, verbose=False)
                fit_obj.lognormal.plot_ccdf(ax=ax, color=color, linestyle="--",
                                             lw=1.0, label="Lognormal")
            except Exception:
                pass

    obs_label = OBSERVABLE_LABELS.get(observable, observable)
    ax.set_xlabel("Event size $x$", fontsize=9)
    ax.set_ylabel("$P(X \\geq x)$", fontsize=9)
    ax.set_title(f"{obs_label}: PL vs lognormal", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig