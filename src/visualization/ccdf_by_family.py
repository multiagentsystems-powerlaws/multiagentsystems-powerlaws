"""
visualization/ccdf_by_family.py
---------------------------------
V2: CCDF by task family.
Tests whether scaling laws arise independent of task semantics.
One panel per task_family (planning, reasoning, coding, qa, synthesis, coordination).
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from visualization._style import apply_style, OBSERVABLE_LABELS
from event_extraction.coordination import (
    extract_delegation_cascades, extract_merge_fanin,
    extract_revision_waves, filter_events, _load_all_events,
)
from tail_fitting.powerlaw_fit import fit_observable

FAMILIES = ["planning", "reasoning", "coding", "qa", "synthesis", "coordination"]
FAMILY_COLORS = {
    "planning":     "#4C72B0",
    "reasoning":    "#DD8452",
    "coding":       "#55A868",
    "qa":           "#C44E52",
    "synthesis":    "#8172B2",
    "coordination": "#DA8BC3",
}
EXTRACTORS = {
    "delegation_cascade": extract_delegation_cascades,
    "revision_wave":      extract_revision_waves,
    "merge_fanin":        extract_merge_fanin,
}

def _ccdf(data):
    x = np.sort(data)
    p = 1.0 - np.arange(len(x)) / len(x)
    return x, p

def plot_ccdf_by_family(
    data_roots:  List[Path],
    out_dir:     Path,
    observable:  str = "delegation_cascade",
    min_samples: int = 20,
    figname:     str = "fig_ccdf_by_family",
):
    apply_style()
    extractor = EXTRACTORS.get(observable, extract_delegation_cascades)
    obs_label = OBSERVABLE_LABELS.get(observable, observable)

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 4.5), squeeze=False)

    for idx, family in enumerate(FAMILIES):
        ax  = axes[idx // ncols][idx % ncols]
        col = FAMILY_COLORS.get(family, "#666")

        sizes: List[float] = []
        for root in data_roots:
            if not root.exists():
                continue
            events   = _load_all_events(root)
            filtered = filter_events(events, task_family=family)
            sizes.extend(extractor(filtered))

        arr = np.array([s for s in sizes if s > 0])
        ax.set_title(family.capitalize(), fontsize=8, pad=3)

        if len(arr) < min_samples:
            ax.text(0.5, 0.5, f"n={len(arr)}\n(need ≥{min_samples})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7, color="gray")
            continue

        x, p = _ccdf(arr)
        ax.loglog(x, p, ".", color=col, alpha=0.4, ms=2, rasterized=True)

        fit = fit_observable(arr, family, verbose=False)
        if fit and fit.n_tail >= min_samples:
            xs = np.logspace(np.log10(fit.x_min), np.log10(arr.max()), 60)
            ys = (xs / fit.x_min) ** (-(fit.alpha - 1))
            xm = np.searchsorted(x, fit.x_min)
            scale = p[xm] if xm < len(p) else 1.0
            ax.loglog(xs, ys * scale, "-", color=col, lw=1.5,
                      label=f"α={fit.alpha:.2f}")
            ax.legend(fontsize=6, frameon=False)

        ax.set_xlabel("Size $x$", fontsize=7)
        ax.set_ylabel("$P(X≥x)$", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"CCDF by task family — {obs_label}", fontsize=9, y=1.01)
    fig.tight_layout()
    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
