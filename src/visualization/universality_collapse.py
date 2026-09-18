"""
visualization/universality_collapse.py
----------------------------------------
Figure 3: Universality collapse plot.
Rescales event sizes by their mean: x' = x / <x>.
If curves from different benchmarks collapse onto one line,
this is evidence of a universal scaling function.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from visualization._style import apply_style, BENCH_COLORS, OBSERVABLE_LABELS
from event_extraction.coordination import (
    extract_delegation_cascades, extract_revision_waves,
    extract_contradiction_bursts, extract_merge_fanin,
    extract_influence_per_agent,
    _load_all_events,
)

_EXTRACTORS = {
    "delegation_cascade":  extract_delegation_cascades,
    "revision_wave":       extract_revision_waves,
    "contradiction_burst": extract_contradiction_bursts,
    "merge_fanin":         extract_merge_fanin,
    "influence_per_agent": extract_influence_per_agent,
}


def _ccdf(data: np.ndarray):
    x = np.sort(data)
    p = 1.0 - np.arange(len(x)) / len(x)
    return x, p


def plot_universality_collapse(
    data_roots:  List[Path],
    out_dir:     Path,
    observable:  str = "delegation_cascade",
    figname:     str = "fig3_universality_collapse",
):
    """
    Rescaled CCDF: x' = x/<x>, P(X'>=x').
    Curves from different benchmarks plotted on same axes.
    Collapse = universality.
    """
    apply_style()
    fig, (ax_raw, ax_scaled) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    obs_label = OBSERVABLE_LABELS.get(observable, observable)

    for root in data_roots:
        if not root.exists():
            continue
        bench = root.name
        color = BENCH_COLORS.get(bench, "#666")
        all_events = _load_all_events(root)
        extractor  = _EXTRACTORS.get(observable, extract_delegation_cascades)
        data       = np.array([s for s in extractor(all_events) if s > 0])

        if len(data) < 20:
            continue

        mean = data.mean()
        x_raw,    p_raw    = _ccdf(data)
        x_scaled, p_scaled = _ccdf(data / mean)

        ax_raw.loglog(x_raw, p_raw, "-", color=color, lw=1.2,
                      alpha=0.8, label=f"{bench} (μ={mean:.1f})")
        ax_scaled.loglog(x_scaled, p_scaled, "-", color=color, lw=1.2,
                         alpha=0.8, label=bench)

    for ax, title, xlabel in [
        (ax_raw,    "Raw CCDF",           "Event size $x$"),
        (ax_scaled, "Rescaled CCDF",      "Rescaled size $x / \\langle x \\rangle$"),
    ]:
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("$P(X \\geq x)$", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"Universality collapse — {obs_label}", fontsize=9)
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig