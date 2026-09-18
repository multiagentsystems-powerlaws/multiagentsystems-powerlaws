"""
visualization/path_length_vs_alpha.py
---------------------------------------
V12: Average path length vs tail exponent.
Tests whether communication constraints influence cascade intensity.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import List
import json
import numpy as np
import matplotlib.pyplot as plt
from visualization._style import apply_style, TOPO_COLORS, TOPO_MARKERS, TOPO_LABELS
from event_extraction.coordination import (
    extract_delegation_cascades, filter_events, _load_all_events,
)
from tail_fitting.powerlaw_fit import fit_observable

def _load_path_length_by_topo(data_roots: List[Path]) -> dict:
    topo_apl: dict = defaultdict(list)
    for root in data_roots:
        if not root.exists():
            continue
        for snap_path in root.rglob("snapshots.jsonl"):
            with open(snap_path) as f:
                for line in f:
                    try:
                        s = json.loads(line.strip())
                        t = s.get("topology", "")
                        v = s.get("average_path_length", None)
                        if t and v and float(v) > 0:
                            topo_apl[t].append(float(v))
                    except Exception:
                        pass
    return topo_apl

def plot_path_length_vs_alpha(
    data_roots: List[Path],
    out_dir:    Path,
    observable: str = "delegation_cascade",
    figname:    str = "fig_path_length_vs_alpha",
):
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    topo_apl = _load_path_length_by_topo(data_roots)

    for topo, apls in topo_apl.items():
        if not apls:
            continue
        sizes: List[float] = []
        for root in data_roots:
            if not root.exists():
                continue
            events   = _load_all_events(root)
            filtered = filter_events(events, topology=topo)
            sizes.extend(extract_delegation_cascades(filtered))

        if len(sizes) < 30:
            continue

        fit = fit_observable(sizes, topo, verbose=False)
        if fit is None:
            continue

        mean_apl = float(np.mean(apls))
        sem_apl  = float(np.std(apls) / max(len(apls)**0.5, 1))
        col      = TOPO_COLORS.get(topo, "#666")
        mrk      = TOPO_MARKERS.get(topo, "o")
        lbl      = TOPO_LABELS.get(topo, topo)

        ax.errorbar(mean_apl, fit.alpha,
                    xerr=sem_apl, yerr=fit.sigma_alpha,
                    fmt=mrk, color=col, ms=7, capsize=3,
                    label=lbl, lw=1.2)

    ax.set_xlabel("Mean average path length $\\ell$", fontsize=9)
    ax.set_ylabel("Scaling exponent $\\alpha$", fontsize=9)
    ax.set_title("Communication distance vs cascade intensity", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # Reference lines
    ax.axhline(3.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1.5, color="gray", lw=0.8, ls=":",  alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 1,
            3.05, "fragile", ha="right", fontsize=6, color="gray")
    ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 1,
            1.55, "collusion risk", ha="right", fontsize=6, color="gray")

    fig.tight_layout()
    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
