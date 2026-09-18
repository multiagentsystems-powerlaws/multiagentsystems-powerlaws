"""
visualization/temporal_burst_plot.py
--------------------------------------
Figure 6 (Visualization 7): Cognitive avalanche raster plot.
X = step, Y = agent, colour = event type.
Vertical clusters = coordination bursts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization._style import apply_style


EVENT_COLORS = {
    "delegate_subtask":  "#4C72B0",
    "complete_subtask":  "#55A868",
    "propose_claim":     "#CCB974",
    "revise_claim":      "#DD8452",
    "contradict_claim":  "#C44E52",
    "merge_claims":      "#8172B2",
    "endorse_claim":     "#DA8BC3",
    "finalize_answer":   "#937860",
}
DEFAULT_COLOR = "#BBBBBB"


def plot_avalanche_raster(
    run_dir:  Path,
    out_dir:  Path,
    figname:  str = "fig6_avalanche_raster",
    max_steps: int = 200,
):
    """
    Raster plot for one run directory.
    Each dot = one agent action at one step.

    Parameters
    ----------
    run_dir   A single run directory containing events.jsonl.
    out_dir   Where to save the figure.
    max_steps Cap on steps shown (truncate long runs).
    """
    apply_style()

    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        print(f"  raster: {events_path} not found, skipping")
        return None

    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    if not events:
        return None

    # Build agent index
    agent_ids  = sorted(set(e["agent_id"] for e in events))
    agent_idx  = {a: i for i, a in enumerate(agent_ids)}

    fig, ax = plt.subplots(figsize=(7.0, max(2.5, len(agent_ids) * 0.35)))

    for ev in events:
        step  = ev.get("step_id", 0)
        aid   = ev.get("agent_id", "")
        etype = ev.get("event_type", "")
        if step > max_steps or aid not in agent_idx:
            continue
        col = EVENT_COLORS.get(etype, DEFAULT_COLOR)
        y   = agent_idx[aid]
        ax.scatter(step, y, c=col, s=12, marker="|", linewidths=1.5,
                   alpha=0.8, zorder=2)

    ax.set_yticks(range(len(agent_ids)))
    ax.set_yticklabels(agent_ids, fontsize=6)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Agent", fontsize=9)
    ax.set_title(
        f"Coordination avalanche raster\n"
        f"{run_dir.parent.parent.name} / {run_dir.parent.name} / {run_dir.name}",
        fontsize=8
    )

    # Legend
    patches = [
        mpatches.Patch(color=c, label=k.replace("_", " "))
        for k, c in EVENT_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=6, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)

    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig


def plot_raster_for_best_run(
    data_roots: List[Path],
    out_dir:    Path,
    topology:   str = "dynamic_reputation",
    figname:    str = "fig6_avalanche_raster",
):
    """Find the run with the most events for a given topology and plot it."""
    best_path  = None
    best_count = 0

    for root in data_roots:
        topo_dir = root / topology
        if not topo_dir.exists():
            continue
        for ep in topo_dir.rglob("events.jsonl"):
            count = sum(1 for _ in open(ep))
            if count > best_count:
                best_count = count
                best_path  = ep.parent

    if best_path is None:
        print(f"  raster: no runs found for topology={topology}")
        return None

    return plot_avalanche_raster(best_path, out_dir, figname)
