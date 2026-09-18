"""
visualization/coalition_timeline.py
--------------------------------------
V10: Coalition evolution timeline.
X = step, Y = agents, colour = coalition membership.
Shows how coalitions form, persist, and dissolve.
Coalition = cluster of agents sharing endorsement/claim lineage within a window.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from visualization._style import apply_style

WINDOW = 10   # steps per coalition detection window

def _detect_coalitions(events: list, window: int = WINDOW):
    """
    Sliding window coalition detection.
    Two agents are in the same coalition if they share a claim lineage
    (same root_claim_id) or endorse each other within the window.
    Returns {step_bin: {agent_id: coalition_id}}.
    """
    if not events:
        return {}
    max_step = max(ev.get("step_id", 0) or 0 for ev in events)
    if max_step == 0:
        return {}

    bins = range(0, max_step + window, window)
    result = {}

    for bin_start in bins:
        bin_end = bin_start + window
        bin_events = [
            ev for ev in events
            if bin_start <= (ev.get("step_id") or 0) < bin_end
        ]
        # Build adjacency: agents sharing root_claim_id are connected
        adj: dict = defaultdict(set)
        for ev in bin_events:
            aid  = ev.get("agent_id", "")
            rcid = ev.get("root_claim_id", "")
            taid = ev.get("target_agent_id", "")
            eid  = ev.get("endorsed_agent_id", "")
            if aid and rcid:
                adj[rcid].add(aid)
            if aid and taid:
                adj[f"pair_{aid}_{taid}"].update([aid, taid])
            if aid and eid:
                adj[f"endorse_{aid}_{eid}"].update([aid, eid])

        # Union-find coalitions
        parent: dict = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            parent[find(x)] = find(y)

        for group in adj.values():
            agents = list(group)
            for i in range(1, len(agents)):
                union(agents[0], agents[i])

        # Map agent → coalition colour id
        all_agents = set(ev.get("agent_id", "") for ev in bin_events if ev.get("agent_id"))
        coalition_map: dict = {}
        coalition_ids: dict = {}
        cid_counter = [0]
        for ag in all_agents:
            root = find(ag)
            if root not in coalition_ids:
                coalition_ids[root] = cid_counter[0]
                cid_counter[0] += 1
            coalition_map[ag] = coalition_ids[root]

        result[bin_start] = coalition_map

    return result


def plot_coalition_timeline(
    data_roots:   List[Path],
    out_dir:      Path,
    topology:     str = "dynamic_reputation",
    figname:      str = "fig_coalition_timeline",
):
    apply_style()

    # Find best run for this topology
    best_path, best_count = None, 0
    for root in data_roots:
        topo_dir = root / topology
        if not topo_dir.exists():
            continue
        for ep in topo_dir.rglob("events.jsonl"):
            count = sum(1 for _ in open(ep))
            if count > best_count:
                best_count = count
                best_path  = ep

    if best_path is None:
        print(f"  coalition_timeline: no runs for topology={topology}")
        return None

    events = []
    with open(best_path) as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except Exception:
                pass

    coalitions = _detect_coalitions(events, WINDOW)
    if not coalitions:
        return None

    all_agents = sorted(set(ev.get("agent_id","") for ev in events if ev.get("agent_id")))
    agent_idx  = {a: i for i, a in enumerate(all_agents)}
    steps      = sorted(coalitions.keys())

    # Build matrix: rows=agents, cols=steps, values=coalition_id
    n_agents = len(all_agents)
    matrix   = np.full((n_agents, len(steps)), -1, dtype=int)
    max_cid  = 0
    for j, step in enumerate(steps):
        cmap = coalitions[step]
        for ag, cid in cmap.items():
            if ag in agent_idx:
                matrix[agent_idx[ag], j] = cid
                max_cid = max(max_cid, cid)

    fig, ax = plt.subplots(figsize=(7.0, max(2.5, n_agents * 0.35)))

    # Use tab20 colourmap, grey for unassigned
    cmap = plt.get_cmap("tab20", max(max_cid + 1, 1))
    masked = np.ma.masked_where(matrix < 0, matrix)
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=max_cid,
              interpolation="nearest",
              extent=[steps[0], steps[-1] + WINDOW, n_agents, 0])

    ax.set_yticks(np.arange(n_agents) + 0.5)
    ax.set_yticklabels(all_agents, fontsize=6)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Agent", fontsize=9)
    ax.set_title(
        f"Coalition evolution — {topology}\n"
        f"(colour = coalition membership, window={WINDOW} steps)",
        fontsize=8
    )
    fig.tight_layout()
    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
