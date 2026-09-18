"""
visualization/cognitive_work_scaling.py
-----------------------------------------
Figure: W ~ S^beta (cognitive work vs cascade size).
Tests whether large events are superlinearly costly.
beta > 1 = synergy, beta < 1 = overhead.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from visualization._style import apply_style, TOPO_COLORS, TOPO_LABELS
from event_extraction.coordination import filter_events, _load_all_events


def _collect_work_vs_size(events: list) -> tuple:
    """
    Returns (sizes, works) arrays for delegation cascades.
    size = agents involved, work = total tokens in cascade.
    """
    subtask_agents: dict = defaultdict(set)
    subtask_tokens: dict = defaultdict(float)

    for ev in events:
        root = ev.get("root_subtask_id") or ev.get("subtask_id")
        if not root:
            continue
        etype = ev.get("event_type", "")
        if etype in ("delegate_subtask", "complete_subtask"):
            subtask_agents[root].add(ev["agent_id"])
            subtask_tokens[root] += ev.get("tokens_total_event") or 0

    sizes = []
    works = []
    for root in subtask_agents:
        s = len(subtask_agents[root])
        w = subtask_tokens[root]
        if s >= 2 and w > 0:
            sizes.append(float(s))
            works.append(float(w))

    return np.array(sizes), np.array(works)


def plot_cognitive_work_scaling(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig_cognitive_work_scaling",
):
    """
    Log-log scatter of cognitive work W vs cascade size S.
    One colour per topology. Fitted line W ~ S^beta.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    all_sizes: List[float] = []
    all_works: List[float] = []

    for root in data_roots:
        if not root.exists():
            continue
        events = _load_all_events(root)

        # Per-topology
        topo_groups: dict = defaultdict(list)
        for ev in events:
            t = ev.get("topology", "unknown")
            topo_groups[t].append(ev)

        for topo, tevents in topo_groups.items():
            sizes, works = _collect_work_vs_size(tevents)
            if len(sizes) < 5:
                continue
            col = TOPO_COLORS.get(topo, "#666")
            ax.scatter(sizes, works, c=col, s=10, alpha=0.4,
                       label=TOPO_LABELS.get(topo, topo), rasterized=True)
            all_sizes.extend(sizes.tolist())
            all_works.extend(works.tolist())

    # Global fit
    if len(all_sizes) >= 10:
        ls = np.log(np.array(all_sizes))
        lw = np.log(np.array(all_works))
        beta, intercept, r, p, se = stats.linregress(ls, lw)
        x_fit = np.logspace(np.log10(min(all_sizes)), np.log10(max(all_sizes)), 60)
        y_fit = np.exp(intercept) * x_fit ** beta
        ax.plot(x_fit, y_fit, "k-", lw=1.8, zorder=5,
                label=f"Fit: β={beta:.2f} (r²={r**2:.2f})")
        ax.text(0.05, 0.92,
                f"β = {beta:.2f}\nβ>1: synergy  β<1: overhead",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cascade size $S$ (agents involved)", fontsize=9)
    ax.set_ylabel("Cognitive work $W$ (tokens)", fontsize=9)
    ax.set_title("Cognitive work scaling $W \\sim S^\\beta$", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
