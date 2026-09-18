"""
visualization/avalanche_raster.py
-----------------------------------
Agent × step heatmap showing coordination burst density.
Different from temporal_burst_plot (scatter) — this is a 2D density heatmap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from visualization._style import apply_style


def plot_avalanche_heatmap(
    data_roots: List[Path],
    out_dir:    Path,
    topology:   str = "dynamic_reputation",
    figname:    str = "fig_avalanche_heatmap",
    n_step_bins: int = 50,
):
    """
    2D heatmap: X = step bins, Y = agent rank (by total events).
    Colour intensity = number of events in that (agent, step_bin) cell.
    """
    apply_style()

    # Collect all (step, agent) pairs for this topology
    agent_step_counts: dict = {}
    max_step = 0

    for root in data_roots:
        topo_dir = root / topology
        if not topo_dir.exists():
            continue
        for ep in topo_dir.rglob("events.jsonl"):
            with open(ep) as f:
                for line in f:
                    try:
                        ev   = json.loads(line.strip())
                        step = ev.get("step_id", 0)
                        aid  = ev.get("agent_id", "")
                        if step and aid:
                            key = (aid, step)
                            agent_step_counts[key] = agent_step_counts.get(key, 0) + 1
                            max_step = max(max_step, step)
                    except Exception:
                        pass

    if not agent_step_counts:
        print(f"  heatmap: no events for topology={topology}")
        return None

    # Build matrix
    all_agents = sorted(set(k[0] for k in agent_step_counts))
    agent_idx  = {a: i for i, a in enumerate(all_agents)}
    step_bins  = np.linspace(0, max_step + 1, n_step_bins + 1)
    bin_idx    = lambda s: int(min(np.searchsorted(step_bins, s) - 1,
                                    n_step_bins - 1))

    matrix = np.zeros((len(all_agents), n_step_bins))
    for (aid, step), count in agent_step_counts.items():
        i = agent_idx[aid]
        j = bin_idx(step)
        matrix[i, j] += count

    # Sort agents by total activity (most active at top)
    order  = np.argsort(matrix.sum(axis=1))[::-1]
    matrix = matrix[order]

    fig, ax = plt.subplots(figsize=(7.0, max(2.5, len(all_agents) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest",
                   extent=[0, max_step, len(all_agents), 0])

    plt.colorbar(im, ax=ax, label="Events per bin", shrink=0.8)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Agent rank (by activity)", fontsize=9)
    ax.set_title(
        f"Coordination burst heatmap — {topology}\n"
        f"({len(all_agents)} agents, {max_step} steps)",
        fontsize=9
    )

    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
