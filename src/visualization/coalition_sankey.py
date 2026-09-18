"""
visualization/coalition_sankey.py
-----------------------------------
Figure: Sankey flow — task type → topology → event type → outcome.
Uses matplotlib for a simplified Sankey (no external library needed).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization._style import apply_style, TOPO_COLORS, BENCH_COLORS


def plot_coalition_sankey(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "fig_coalition_sankey",
):
    """
    Simplified Sankey: horizontal flow diagram showing
    benchmark → topology → dominant event type → success/fail.

    Uses proportional bar segments (matplotlib, no extra dependencies).
    """
    apply_style()

    # Collect flow data
    flows: dict = defaultdict(int)  # (bench, topo, etype, outcome) -> count

    for root in data_roots:
        if not root.exists():
            continue
        bench = root.name
        for meta_path in root.rglob("run_metadata.json"):
            run_dir = meta_path.parent
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue

            score   = meta.get("task_score") or 0.0
            outcome = "success" if float(score) > 0.5 else "fail"

            events_path = run_dir / "events.jsonl"
            if not events_path.exists():
                continue

            topo   = "unknown"
            etypes = Counter()
            with open(events_path) as f:
                for line in f:
                    try:
                        ev = json.loads(line.strip())
                        if ev.get("topology"):
                            topo = ev["topology"]
                        if ev.get("event_type"):
                            etypes[ev["event_type"]] += 1
                    except Exception:
                        pass

            etype = etypes.most_common(1)[0][0] if etypes else "unknown"
            flows[(bench, topo, etype, outcome)] += 1

    if not flows:
        print("  sankey: no flow data, skipping")
        return None

    # Aggregate by each stage
    bench_counts: Counter  = Counter()
    topo_counts:  Counter  = Counter()
    etype_counts: Counter  = Counter()
    outcome_counts: Counter = Counter()

    for (bench, topo, etype, outcome), count in flows.items():
        bench_counts[bench]   += count
        topo_counts[topo]     += count
        etype_counts[etype]   += count
        outcome_counts[outcome] += count

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 4.0),
                              gridspec_kw={"wspace": 0.1})
    titles  = ["Benchmark", "Topology", "Event type", "Outcome"]
    counters = [bench_counts, topo_counts, etype_counts, outcome_counts]

    COLOR_MAPS = [BENCH_COLORS, TOPO_COLORS, {}, {"success":"#4CAF50","fail":"#F44336"}]

    for ax, title, counter, cmap in zip(axes, titles, counters, COLOR_MAPS):
        keys   = sorted(counter.keys(), key=lambda k: -counter[k])[:8]
        vals   = np.array([counter[k] for k in keys], dtype=float)
        vals  /= vals.sum()
        colors = [cmap.get(k, "#AAAAAA") for k in keys]

        # Horizontal stacked bar
        left = 0.0
        for k, v, c in zip(keys, vals, colors):
            ax.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", lw=0.5)
            if v > 0.05:
                ax.text(left + v / 2, 0,
                        k.replace("_","\n"), ha="center", va="center",
                        fontsize=5.5, color="white" if v > 0.12 else "#333")
            left += v

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle("Flow: benchmark → topology → event type → outcome", fontsize=8)
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
