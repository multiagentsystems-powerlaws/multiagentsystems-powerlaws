"""
visualization/lr_test_heatmap.py
---------------------------------
Figure 4 (Visualization 4): Likelihood-ratio test results as a heatmap.
Rows = observables, Columns = alternative distributions.
Cell colour = LR statistic (positive = PL wins).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from visualization._style import apply_style, OBSERVABLE_LABELS
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
from tail_fitting.powerlaw_fit import fit_all, FitResult


def plot_lr_heatmap(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig4_lr_heatmap",
):
    """
    Heatmap of LR statistics: power-law vs lognormal, exponential, truncated PL.
    Positive = power law wins. Starred cells = statistically significant (p<0.05).
    """
    apply_style()

    # Pool all data across roots
    pooled: Dict[str, List[float]] = {}
    for root in data_roots:
        if not root.exists():
            continue
        all_events = _load_all_events(root)
        for name, fn in _EXTRACTORS.items():
            pooled.setdefault(name, []).extend(fn(all_events))

    fits = fit_all(pooled, verbose=False)
    if not fits:
        print("  lr_test_heatmap: no fitted observables, skipping")
        return None

    obs_names  = list(fits.keys())
    alt_names  = ["vs Lognormal", "vs Exponential", "vs Truncated PL"]
    lr_matrix  = np.zeros((len(obs_names), len(alt_names)))
    sig_matrix = np.zeros((len(obs_names), len(alt_names)), dtype=bool)

    for i, name in enumerate(obs_names):
        r = fits[name]
        lr_matrix[i, 0]  = r.lr_vs_lognormal
        lr_matrix[i, 1]  = r.lr_vs_exponential
        lr_matrix[i, 2]  = r.lr_vs_truncated_pl
        sig_matrix[i, 0] = r.lr_vs_lognormal_p < 0.05
        sig_matrix[i, 1] = r.lr_vs_exponential_p < 0.05
        sig_matrix[i, 2] = r.lr_vs_truncated_pl_p < 0.05

    obs_labels = [OBSERVABLE_LABELS.get(n, n) for n in obs_names]

    vmax = max(abs(lr_matrix).max(), 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 0.55 * len(obs_names) + 1.2))

    im = ax.imshow(
        lr_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"
    )

    ax.set_xticks(range(len(alt_names)))
    ax.set_xticklabels(alt_names, fontsize=8)
    ax.set_yticks(range(len(obs_labels)))
    ax.set_yticklabels(obs_labels, fontsize=8)
    ax.set_title("Likelihood-ratio: power law vs alternatives\n"
                 "(positive = PL wins, * = p<0.05)", fontsize=9)

    # Annotate cells
    for i in range(len(obs_names)):
        for j in range(len(alt_names)):
            val  = lr_matrix[i, j]
            star = "*" if sig_matrix[i, j] else ""
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}{star}",
                    ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("LR statistic", fontsize=8)
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig