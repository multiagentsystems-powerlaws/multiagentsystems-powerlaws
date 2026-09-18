"""
visualization/event_vs_success.py
-----------------------------------
Figure: Event size vs task success probability.
Tests whether large coordination cascades actually solve harder tasks.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from visualization._style import apply_style, TOPO_COLORS, TOPO_LABELS


def plot_event_vs_success(
    data_roots: List[Path],
    out_dir:    Path,
    n_bins:     int = 8,
    figname:    str = "fig_event_vs_success",
):
    """
    Scatter + binned mean: max event size per run vs task success score.
    Shows whether larger coordination events correlate with better outcomes.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    ax_scatter, ax_binned = axes

    all_sizes:   List[float] = []
    all_success: List[float] = []

    for root in data_roots:
        if not root.exists():
            continue
        for meta_path in root.rglob("run_metadata.json"):
            run_dir = meta_path.parent
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue

            score = meta.get("task_score")
            if score is None:
                continue

            events_path = run_dir / "events.jsonl"
            if not events_path.exists():
                continue

            # Max event size = max merge_num_inputs or cascade size proxy
            max_size = 0
            topo     = "unknown"
            with open(events_path) as f:
                for line in f:
                    try:
                        ev = json.loads(line.strip())
                        n  = ev.get("merge_num_inputs") or ev.get("num_agents_involved") or 0
                        if n > max_size:
                            max_size = n
                        if ev.get("topology"):
                            topo = ev["topology"]
                    except Exception:
                        pass

            if max_size > 0:
                col = TOPO_COLORS.get(topo, "#666")
                ax_scatter.scatter(max_size, float(score),
                                   c=col, s=12, alpha=0.4, rasterized=True)
                all_sizes.append(float(max_size))
                all_success.append(float(score))

    if len(all_sizes) >= 10:
        sizes   = np.array(all_sizes)
        success = np.array(all_success)

        # Binned mean ± SE
        bin_means, edges, _ = binned_statistic(sizes, success, "mean",   bins=n_bins)
        bin_stds,  _,    _  = binned_statistic(sizes, success, "std",    bins=n_bins)
        bin_counts,_,    _  = binned_statistic(sizes, success, "count",  bins=n_bins)
        bin_se = bin_stds / np.sqrt(np.maximum(bin_counts, 1))
        centres = 0.5 * (edges[:-1] + edges[1:])

        valid = ~np.isnan(bin_means)
        ax_binned.errorbar(
            centres[valid], bin_means[valid], yerr=bin_se[valid],
            fmt="o-", color="#333", lw=1.5, ms=5, capsize=3,
        )

        # Linear trend
        from scipy.stats import linregress
        slope, intercept, r, p, _ = linregress(sizes, success)
        x_fit = np.linspace(sizes.min(), sizes.max(), 50)
        ax_binned.plot(x_fit, intercept + slope * x_fit,
                       "--", color="gray", lw=1.0)
        ax_binned.text(0.05, 0.92,
                       f"slope={slope:.3f}  p={p:.3f}",
                       transform=ax_binned.transAxes, fontsize=7,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    for ax, title in [(ax_scatter, "Raw scatter"),
                       (ax_binned,  "Binned mean ± SE")]:
        ax.set_xlabel("Max event size per run", fontsize=9)
        ax.set_ylabel("Task success score", fontsize=9)
        ax.set_title(title, fontsize=8)

    fig.suptitle("Event size vs task success", fontsize=9, y=1.01)
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
