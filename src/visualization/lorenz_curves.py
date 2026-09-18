"""
visualization/lorenz_curves.py
--------------------------------
Lorenz curves of agent influence per topology.
Shows inequality in who drives collective reasoning.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from visualization._style import apply_style, TOPO_COLORS, TOPO_LABELS
from event_extraction.coordination import (
    extract_influence_per_agent, filter_events, _load_all_events
)


def _lorenz(arr: np.ndarray):
    """Compute Lorenz curve (x=cumulative agents, y=cumulative influence)."""
    arr = np.sort(arr.astype(float))
    x   = np.linspace(0, 1, len(arr) + 1)
    y   = np.concatenate([[0], np.cumsum(arr) / arr.sum()])
    return x, y


def _gini_from_lorenz(x, y) -> float:
    return float(1 - 2 * np.trapz(y, x))


def plot_lorenz_curves(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig_lorenz_curves",
):
    """
    Lorenz curves of agent influence, one curve per topology.
    Diagonal = perfect equality. Bowed curve = concentration.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    # Equality line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Equality")

    topos_found = []
    for root in data_roots:
        if not root.exists():
            continue
        events = _load_all_events(root)

        topo_groups: dict = {}
        for ev in events:
            t = ev.get("topology", "unknown")
            topo_groups.setdefault(t, []).append(ev)

        for topo, tevents in topo_groups.items():
            scores = extract_influence_per_agent(tevents)
            if len(scores) < 4:
                continue
            arr      = np.array(scores)
            x, y     = _lorenz(arr)
            gini_val = _gini_from_lorenz(x, y)
            col      = TOPO_COLORS.get(topo, "#666")
            lbl      = f"{TOPO_LABELS.get(topo, topo)} (G={gini_val:.2f})"
            if topo not in topos_found:
                ax.plot(x, y, "-", color=col, lw=1.5, label=lbl)
                ax.fill_between(x, y, x, alpha=0.05, color=col)
                topos_found.append(topo)

    ax.set_xlabel("Cumulative fraction of agents", fontsize=9)
    ax.set_ylabel("Cumulative fraction of influence", fontsize=9)
    ax.set_title("Lorenz curves of agent influence\nby topology", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
