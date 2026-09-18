"""
visualization/avalanche_size_duration.py
------------------------------------------
V8: Avalanche size vs duration S ~ T^gamma.
Tests dynamical scaling between event size and duration.
Duration = number of distinct steps spanned by an event cluster.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import List
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from visualization._style import apply_style, TOPO_COLORS, TOPO_LABELS

def _extract_cascade_size_duration(events: list):
    """
    For each delegation subtask cascade:
      size     = distinct agents involved
      duration = max(step_id) - min(step_id) within cascade
    Returns (sizes, durations).
    """
    subtask_agents: dict = defaultdict(set)
    subtask_steps:  dict = defaultdict(list)
    for ev in events:
        if ev.get("event_type") in ("delegate_subtask", "complete_subtask"):
            root = ev.get("root_subtask_id") or ev.get("subtask_id", "")
            if root:
                subtask_agents[root].add(ev.get("agent_id", ""))
                step = ev.get("step_id")
                if step is not None:
                    subtask_steps[root].append(int(step))
    sizes, durations = [], []
    for root in subtask_agents:
        s = len(subtask_agents[root])
        steps = subtask_steps.get(root, [])
        d = max(steps) - min(steps) + 1 if len(steps) >= 2 else 1
        if s >= 2 and d >= 1:
            sizes.append(float(s))
            durations.append(float(d))
    return np.array(sizes), np.array(durations)

def plot_avalanche_size_duration(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig_avalanche_size_duration",
):
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    all_sizes, all_durs = [], []
    for root in data_roots:
        if not root.exists():
            continue
        for topo_dir in sorted(root.iterdir()):
            if not topo_dir.is_dir():
                continue
            topo = topo_dir.name
            col  = TOPO_COLORS.get(topo, "#AAAAAA")
            topo_sizes, topo_durs = [], []
            for ep in topo_dir.rglob("events.jsonl"):
                events = []
                with open(ep) as f:
                    for line in f:
                        try:
                            events.append(json.loads(line.strip()))
                        except Exception:
                            pass
                s, d = _extract_cascade_size_duration(events)
                topo_sizes.extend(s.tolist())
                topo_durs.extend(d.tolist())

            if len(topo_sizes) >= 5:
                ax.scatter(topo_durs, topo_sizes, c=col, s=8, alpha=0.4,
                           label=TOPO_LABELS.get(topo, topo), rasterized=True)
                all_sizes.extend(topo_sizes)
                all_durs.extend(topo_durs)

    if len(all_sizes) >= 10:
        ld = np.log(np.maximum(all_durs, 1))
        ls = np.log(np.maximum(all_sizes, 1))
        gamma, intercept, r, p, _ = stats.linregress(ld, ls)
        d_fit = np.logspace(np.log10(max(min(all_durs), 1)),
                             np.log10(max(all_durs)), 50)
        s_fit = np.exp(intercept) * d_fit ** gamma
        ax.plot(d_fit, s_fit, "k-", lw=2, zorder=5,
                label=f"S ~ T^γ  γ={gamma:.2f} (r²={r**2:.2f})")
        ax.text(0.05, 0.92,
                f"γ = {gamma:.2f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Avalanche duration $T$ (steps)", fontsize=9)
    ax.set_ylabel("Avalanche size $S$ (agents)", fontsize=9)
    ax.set_title("Dynamical scaling: $S \\sim T^\\gamma$", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
