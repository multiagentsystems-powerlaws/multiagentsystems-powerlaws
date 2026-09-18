"""
visualization/topology_comparison.py
--------------------------------------
Side-by-side comparison of fitted alpha and tail shape across all 7 topologies.
Bar chart + error bars + regime colour coding.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization._style import apply_style, TOPO_COLORS, TOPO_LABELS
from event_extraction.coordination import (
    extract_delegation_cascades, filter_events, _load_all_events
)
from tail_fitting.powerlaw_fit import fit_observable

REGIME_COLORS = {
    "collusion_risk":         "#F44336",
    "collective_intelligence":"#4CAF50",
    "fragile_reasoning":      "#9E9E9E",
}


def plot_topology_comparison(
    data_roots: List[Path],
    out_dir:    Path,
    observable: str = "delegation_cascade",
    figname:    str = "fig_topology_comparison",
):
    """Bar chart of fitted alpha ± sigma per topology."""
    apply_style()

    alphas: dict  = {}
    sigmas: dict  = {}
    regimes: dict = {}
    ntails: dict  = {}

    for root in data_roots:
        if not root.exists():
            continue
        events = _load_all_events(root)
        topo_groups: dict = {}
        for ev in events:
            t = ev.get("topology", "")
            if t:
                topo_groups.setdefault(t, []).append(ev)

        for topo, tevents in topo_groups.items():
            sizes = extract_delegation_cascades(tevents)
            if len(sizes) < 30:
                continue
            fit = fit_observable(sizes, topo, verbose=False)
            if fit is None:
                continue
            if topo not in alphas or fit.n_tail > ntails.get(topo, 0):
                alphas[topo]  = fit.alpha
                sigmas[topo]  = fit.sigma_alpha
                regimes[topo] = fit.regime
                ntails[topo]  = fit.n_tail

    if not alphas:
        print("  topology_comparison: no fitted topologies, skipping")
        return None

    topos = sorted(alphas.keys())
    x     = np.arange(len(topos))
    vals  = [alphas[t]  for t in topos]
    errs  = [sigmas[t]  for t in topos]
    cols  = [REGIME_COLORS.get(regimes.get(t,""), "#666") for t in topos]
    xlbls = [TOPO_LABELS.get(t, t) for t in topos]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(x, vals, yerr=errs, color=cols, capsize=4,
                  alpha=0.85, edgecolor="white", lw=0.5)

    # Annotate n_tail
    for i, t in enumerate(topos):
        ax.text(i, vals[i] + errs[i] + 0.05, f"n={ntails[t]}",
                ha="center", va="bottom", fontsize=6, color="#555")

    ax.axhline(3.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(1.5, color="gray", lw=0.8, ls=":",  alpha=0.6)
    ax.text(len(topos) - 0.3, 3.05, "fragile", ha="right", fontsize=6, color="gray")
    ax.text(len(topos) - 0.3, 1.55, "collusion", ha="right", fontsize=6, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Scaling exponent $\\alpha$", fontsize=9)
    ax.set_title("Power-law exponent by topology", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=c, label=k.replace("_", " "))
        for k, c in REGIME_COLORS.items()
    ]
    ax.legend(handles=legend_patches, fontsize=7, frameon=False)

    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
