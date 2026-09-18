"""
visualization/mean_vs_max_scaling.py
--------------------------------------
Figure: Mean vs max event size as a function of N agents.
Tests whether systems show Zipf-like extreme events: x_max >> <x>.

Two panels:
  Left:  <x> vs N  (mean cascade size scales with N)
  Right: x_max / <x> vs N  (ratio grows = heavier tail at larger N)

If x_max / <x> grows with N, this is evidence of H3:
coordination events become more extreme at scale.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from visualization._style import (
    apply_style, TOPO_COLORS, TOPO_MARKERS, TOPO_LABELS,
)
from event_extraction.coordination import (
    extract_delegation_cascades, extract_revision_waves,
    extract_merge_fanin, extract_influence_per_agent,
    _load_all_events,
)

_EXTRACTORS = {
    "delegation_cascade":  extract_delegation_cascades,
    "revision_wave":       extract_revision_waves,
    "merge_fanin":         extract_merge_fanin,
    "influence_per_agent": extract_influence_per_agent,
}


def _collect_by_n(
    data_roots: List[Path],
    observable: str,
    topology:   Optional[str] = None,
) -> Dict[int, List[float]]:
    """
    Returns {num_agents: [all event sizes across seeds/tasks at that N]}.
    Walks the {data-root}/{topology}/n{N}/{seed}/{task}/ directory structure.
    """
    extractor  = _EXTRACTORS.get(observable, extract_delegation_cascades)
    n_to_sizes: Dict[int, List[float]] = defaultdict(list)

    for root in data_roots:
        if not root.exists():
            continue
        for topo_dir in sorted(root.iterdir()):
            if not topo_dir.is_dir():
                continue
            if topology and topo_dir.name != topology:
                continue
            for n_dir in sorted(topo_dir.iterdir()):
                if not n_dir.is_dir() or not n_dir.name.startswith("n"):
                    continue
                try:
                    n = int(n_dir.name[1:])
                except ValueError:
                    continue
                for seed_dir in n_dir.iterdir():
                    if not seed_dir.is_dir():
                        continue
                    for task_dir in seed_dir.iterdir():
                        ep = task_dir / "events.jsonl"
                        if not ep.exists():
                            continue
                        events = []
                        with open(ep) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        events.append(json.loads(line))
                                    except Exception:
                                        pass
                        sizes = extractor(events)
                        n_to_sizes[n].extend(sizes)

    return n_to_sizes


def plot_mean_vs_max_scaling(
    data_roots:  List[Path],
    out_dir:     Path,
    observable:  str = "delegation_cascade",
    topologies:  Optional[List[str]] = None,
    figname:     str = "fig_mean_vs_max_scaling",
):
    """
    Two-panel figure:
      Left:  mean event size <x> vs N
      Right: x_max / <x> vs N (extreme event ratio)

    One line per topology. Log-log axes. OLS fits annotated.
    """
    apply_style()
    fig, (ax_mean, ax_ratio) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    topos = topologies or list(TOPO_COLORS.keys())
    any_data = False

    for topo in topos:
        col = TOPO_COLORS.get(topo, "#666")
        mrk = TOPO_MARKERS.get(topo, "o")
        lbl = TOPO_LABELS.get(topo, topo)

        n_to_sizes = _collect_by_n(data_roots, observable, topo)
        if len(n_to_sizes) < 2:
            continue

        Ns     = np.array(sorted(n_to_sizes.keys()))
        means  = np.array([np.mean(n_to_sizes[n]) for n in Ns])
        maxes  = np.array([np.max(n_to_sizes[n])  for n in Ns])
        ratios = maxes / np.maximum(means, 1e-9)

        # SEM for means
        sems_mean = np.array([
            np.std(n_to_sizes[n]) / max(len(n_to_sizes[n])**0.5, 1)
            for n in Ns
        ])

        ax_mean.errorbar(Ns, means, yerr=sems_mean,
                         fmt=f"{mrk}-", color=col, lw=1.2,
                         ms=4, capsize=2, label=lbl)
        ax_ratio.plot(Ns, ratios, f"{mrk}-", color=col, lw=1.2,
                      ms=4, label=lbl)

        # Fit slopes
        if len(Ns) >= 3:
            log_n = np.log(Ns)
            # Mean slope
            lm    = np.log(np.maximum(means, 1e-9))
            bm, im, rm, pm, _ = stats.linregress(log_n, lm)
            xf   = np.linspace(Ns.min(), Ns.max(), 40)
            ax_mean.plot(xf, np.exp(im) * xf**bm, "--",
                         color=col, lw=0.8, alpha=0.5)
            ax_mean.annotate(f"β={bm:.2f}",
                             xy=(Ns[-1], means[-1]),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=6, color=col, va="center")
            # Ratio slope
            lr_   = np.log(np.maximum(ratios, 1e-9))
            br, ir, rr, pr, _ = stats.linregress(log_n, lr_)
            ax_ratio.plot(xf, np.exp(ir) * xf**br, "--",
                          color=col, lw=0.8, alpha=0.5)
            ax_ratio.annotate(f"γ={br:.2f}",
                              xy=(Ns[-1], ratios[-1]),
                              xytext=(4, 0), textcoords="offset points",
                              fontsize=6, color=col, va="center")

        any_data = True

    if not any_data:
        print("  mean_vs_max_scaling: no multi-scale data found "
              "(need --scales 8 16 32 64 128 in sweep)")
        plt.close(fig)
        return None

    for ax, title, ylabel in [
        (ax_mean,  r"Mean event size $\langle x \rangle$ vs $N$",
                   r"$\langle x \rangle$"),
        (ax_ratio, r"Extreme event ratio $x_{\max}/\langle x \rangle$ vs $N$",
                   r"$x_{\max} / \langle x \rangle$"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of agents $N$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, frameon=False, ncol=2)

    fig.suptitle("H3: Scaling of coordination events with system size",
                 fontsize=9)
    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig