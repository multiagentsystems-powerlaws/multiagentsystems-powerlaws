"""
visualization/mega_cascade.py
-------------------------------
Figure: Mega cascade case study — full visualisation of the single largest
delegation cascade found across all runs.
Shows: delegation tree structure + revision points + merge fan-in.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from visualization._style import apply_style


def _find_largest_cascade(data_roots: List[Path]) -> Tuple[Optional[Path], str]:
    """Find the run_dir and root_subtask_id with the most delegation events."""
    best_path  = None
    best_root  = ""
    best_count = 0

    for root in data_roots:
        if not root.exists():
            continue
        for ep in root.rglob("events.jsonl"):
            subtask_counts: dict = defaultdict(int)
            with open(ep) as f:
                for line in f:
                    try:
                        ev = json.loads(line.strip())
                        if ev.get("event_type") in ("delegate_subtask","complete_subtask"):
                            rk = ev.get("root_subtask_id") or ev.get("subtask_id","")
                            if rk:
                                subtask_counts[rk] += 1
                    except Exception:
                        pass
            if subtask_counts:
                top_root, top_count = max(subtask_counts.items(), key=lambda x: x[1])
                if top_count > best_count:
                    best_count = top_count
                    best_path  = ep.parent
                    best_root  = top_root

    return best_path, best_root


def plot_mega_cascade(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig_mega_cascade",
):
    """
    Find the largest cascade in all data and visualise it as a tree.
    Nodes coloured by event type, sized by tokens used.
    """
    apply_style()

    run_dir, root_subtask = _find_largest_cascade(data_roots)
    if run_dir is None:
        print("  mega_cascade: no delegation cascades found, skipping")
        return None

    # Collect all events in this cascade
    cascade_events = []
    events_path    = run_dir / "events.jsonl"
    with open(events_path) as f:
        for line in f:
            try:
                ev = json.loads(line.strip())
                rk = ev.get("root_subtask_id") or ev.get("subtask_id", "")
                if rk == root_subtask:
                    cascade_events.append(ev)
            except Exception:
                pass

    if not cascade_events:
        print(f"  mega_cascade: no events for root={root_subtask}")
        return None

    # Build subtask tree
    G       = nx.DiGraph()
    node_ev: dict = {}
    for ev in cascade_events:
        sid = ev.get("subtask_id", ev.get("claim_id", ""))
        pid = ev.get("parent_subtask_id", "")
        if not sid:
            continue
        G.add_node(sid)
        node_ev[sid] = ev
        if pid and pid != sid:
            G.add_edge(pid, sid)

    if G.number_of_nodes() == 0:
        print("  mega_cascade: empty tree")
        return None

    ETYPE_COLORS = {
        "delegate_subtask":  "#4C72B0",
        "complete_subtask":  "#55A868",
        "merge_claims":      "#8172B2",
        "revise_claim":      "#DD8452",
        "propose_claim":     "#CCB974",
    }

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)

    node_colors = [
        ETYPE_COLORS.get(node_ev.get(n, {}).get("event_type",""), "#CCCCCC")
        for n in G.nodes()
    ]
    node_sizes = [
        max(50, min(400, (node_ev.get(n, {}).get("tokens_total_event") or 50) * 0.5))
        for n in G.nodes()
    ]

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=False,
        edge_color="#CCCCCC",
        arrows=True, arrowsize=7,
        width=0.6, alpha=0.9,
    )

    patches = [mpatches.Patch(color=c, label=k.replace("_"," "))
               for k, c in ETYPE_COLORS.items()]
    ax.legend(handles=patches, fontsize=6, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    topo = cascade_events[0].get("topology", "?")
    ax.set_title(
        f"Mega cascade — {topo}\n"
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} links  "
        f"(root: {root_subtask[:16]}...)",
        fontsize=8
    )
    ax.axis("off")
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
