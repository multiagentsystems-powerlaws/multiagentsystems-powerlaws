"""
visualization/regime_phase_diagram.py
---------------------------------------
Figure 8 (H2): MAS phase space map.
X = influence concentration (Gini), Y = coalition persistence.
Colour = task success, size = bridge traffic.
Clusters identify: cooperation / competition / collusion / coopetition.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from visualization._style import apply_style, TOPO_COLORS, TOPO_MARKERS, TOPO_LABELS


def _gini(arr: np.ndarray) -> float:
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr.astype(float))
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum()) - (n + 1) / n))


def _load_run_metrics(data_roots: List[Path]) -> list:
    """
    Load per-run metrics from run_metadata.json + events.jsonl.
    Returns list of dicts with gini, coalition_persistence, success, bridge_traffic, topology.
    """
    runs = []
    for root in data_roots:
        if not root.exists():
            continue
        for meta_path in root.rglob("run_metadata.json"):
            run_dir = meta_path.parent
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue

            # Load events for this run
            events_path = run_dir / "events.jsonl"
            if not events_path.exists():
                continue

            events = []
            with open(events_path) as f:
                for line in f:
                    try:
                        events.append(json.loads(line.strip()))
                    except Exception:
                        pass

            if not events:
                continue

            # Gini of influence
            influence: dict = defaultdict(float)
            for ev in events:
                influence[ev.get("agent_id","")] += 1
            gini = _gini(np.array(list(influence.values())))

            # Coalition persistence proxy: fraction of endorse_claim events
            n_total   = len(events)
            n_endorse = sum(1 for e in events if e.get("event_type") == "endorse_claim")
            coalition_persistence = n_endorse / max(n_total, 1)

            # Bridge traffic: fraction of events involving bridge agents
            n_bridge = sum(1 for e in events if e.get("agent_role") == "bridge")
            bridge_traffic = n_bridge / max(n_total, 1)

            # Topology from first event
            topology = events[0].get("topology", "unknown") if events else "unknown"

            # Success from metadata
            success = meta.get("task_score") or 0.0

            runs.append({
                "gini":                  gini,
                "coalition_persistence": coalition_persistence,
                "bridge_traffic":        bridge_traffic,
                "success":               float(success),
                "topology":              topology,
                "run_id":                meta.get("run_id", ""),
            })

    return runs


def plot_regime_phase_diagram(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig8_regime_phase_diagram",
):
    """
    Scatter plot: Gini × coalition persistence.
    Colour = task success (0–1), size = bridge traffic.
    """
    apply_style()
    runs = _load_run_metrics(data_roots)

    if not runs:
        print("  phase_diagram: no run data found")
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ginis        = np.array([r["gini"]                  for r in runs])
    persistence  = np.array([r["coalition_persistence"]  for r in runs])
    successes    = np.array([r["success"]                for r in runs])
    bridge_sizes = np.array([r["bridge_traffic"]         for r in runs])
    topologies   = [r["topology"] for r in runs]

    # Normalise sizes: 30–300
    sz = 30 + 270 * (bridge_sizes / max(bridge_sizes.max(), 1e-9))

    cmap  = plt.get_cmap("RdYlGn")
    norm  = Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(s)) for s in successes]

    sc = ax.scatter(ginis, persistence, c=successes, s=sz,
                    cmap="RdYlGn", vmin=0, vmax=1,
                    alpha=0.75, edgecolors="white", lw=0.4, zorder=3)

    # Per-topology mean markers
    for topo in set(topologies):
        mask = np.array([t == topo for t in topologies])
        if mask.sum() < 2:
            continue
        ax.plot(ginis[mask].mean(), persistence[mask].mean(),
                marker=TOPO_MARKERS.get(topo,"o"),
                color=TOPO_COLORS.get(topo,"#666"),
                ms=10, mew=1.5, mec="white", zorder=5,
                label=TOPO_LABELS.get(topo, topo))

    # Regime labels
    ax.text(0.08, 0.88, "Cooperation",   transform=ax.transAxes,
            fontsize=7, color="#2196F3", style="italic")
    ax.text(0.65, 0.88, "Coopetition",   transform=ax.transAxes,
            fontsize=7, color="#FF9800", style="italic")
    ax.text(0.08, 0.08, "Competition",   transform=ax.transAxes,
            fontsize=7, color="#9C27B0", style="italic")
    ax.text(0.65, 0.08, "Collusion",     transform=ax.transAxes,
            fontsize=7, color="#F44336", style="italic")

    ax.axvline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(0.1, color="gray", lw=0.7, ls="--", alpha=0.5)

    plt.colorbar(sc, ax=ax, label="Task success score", shrink=0.85)
    ax.set_xlabel("Influence concentration (Gini)", fontsize=9)
    ax.set_ylabel("Coalition persistence", fontsize=9)
    ax.set_title("MAS phase space map", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2,
              bbox_to_anchor=(0.5, -0.18), loc="upper center")

    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
