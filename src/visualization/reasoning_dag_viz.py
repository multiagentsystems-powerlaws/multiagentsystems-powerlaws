"""
visualization/reasoning_dag_viz.py
------------------------------------
Figure: Reasoning DAG for one run.
Nodes = claims, edges = delegation/revision/merge/endorse.
Coloured by event type, sized by agent influence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from visualization._style import apply_style


NODE_COLORS = {
    "propose_claim":    "#4C72B0",
    "revise_claim":     "#DD8452",
    "contradict_claim": "#C44E52",
    "merge_claims":     "#8172B2",
    "endorse_claim":    "#DA8BC3",
    "delegate_subtask": "#55A868",
}


def plot_reasoning_dag(
    run_dir:    Path,
    out_dir:    Path,
    figname:    str = "fig_reasoning_dag",
    max_nodes:  int = 60,
):
    """
    Build and plot the claim DAG for one run.

    Nodes = claims (one per event with a claim_id).
    Edges = parent_claim_ids → claim_id.
    """
    apply_style()

    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        print(f"  dag: {events_path} not found")
        return None

    events = []
    with open(events_path) as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except Exception:
                pass

    # Build DAG
    G = nx.DiGraph()
    node_color_map: dict = {}
    node_agent_map: dict = {}

    # Influence scores
    influence: dict = {}
    for ev in events:
        aid = ev.get("agent_id", "")
        influence[aid] = influence.get(aid, 0) + 1

    for ev in events:
        cid = ev.get("claim_id")
        if not cid:
            continue
        etype  = ev.get("event_type", "propose_claim")
        aid    = ev.get("agent_id", "")
        parents = ev.get("parent_claim_ids", [])

        if len(G) >= max_nodes:
            break

        G.add_node(cid, event_type=etype, agent=aid)
        node_color_map[cid] = NODE_COLORS.get(etype, "#CCCCCC")
        node_agent_map[cid] = aid

        for pid in (parents or []):
            if pid and pid != cid:
                G.add_edge(pid, cid)

        # Also add merge parents
        for pid in ev.get("merge_parent_claim_ids", []):
            if pid and pid != cid:
                G.add_node(pid)
                G.add_edge(pid, cid)

    if G.number_of_nodes() == 0:
        print("  dag: no claim nodes found")
        return None

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Layout: try hierarchical, fall back to spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.0)

    node_colors = [node_color_map.get(n, "#CCCCCC") for n in G.nodes()]
    node_sizes  = [
        50 + 20 * influence.get(node_agent_map.get(n, ""), 0)
        for n in G.nodes()
    ]

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=False,
        edge_color="#AAAAAA",
        arrows=True,
        arrowsize=8,
        width=0.7,
        alpha=0.85,
    )

    patches = [
        mpatches.Patch(color=c, label=k.replace("_", " "))
        for k, c in NODE_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=6, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    ax.set_title(
        f"Reasoning DAG — {run_dir.parent.parent.name} / "
        f"{run_dir.parent.name} / {run_dir.name}\n"
        f"({G.number_of_nodes()} claims, {G.number_of_edges()} links)",
        fontsize=8
    )
    ax.axis("off")
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
