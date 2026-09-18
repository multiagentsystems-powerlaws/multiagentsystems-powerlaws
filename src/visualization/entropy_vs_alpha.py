"""
visualization/entropy_vs_alpha.py
----------------------------------
Figure: Graph entropy H(G) vs power-law exponent alpha.
Tests whether structural flexibility drives cascade dynamics.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from visualization._style import apply_style, TOPO_COLORS, TOPO_MARKERS, TOPO_LABELS
from tail_fitting.powerlaw_fit import fit_observable, FitResult
from event_extraction.coordination import (
    extract_delegation_cascades, _load_all_events, filter_events
)


def _load_entropy_by_topo(data_roots: List[Path]) -> dict:
    """Load mean graph_entropy per topology from snapshots.jsonl files."""
    topo_entropy: dict = defaultdict(list)
    for root in data_roots:
        if not root.exists():
            continue
        for snap_path in root.rglob("snapshots.jsonl"):
            with open(snap_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                        topo = s.get("topology", "")
                        h    = s.get("graph_entropy", None)
                        if topo and h is not None and h > 0:
                            topo_entropy[topo].append(float(h))
                    except Exception:
                        pass
    return topo_entropy


def plot_entropy_vs_alpha(
    data_roots: List[Path],
    out_dir:    Path,
    observable: str = "delegation_cascade",
    figname:    str = "fig_entropy_vs_alpha",
):
    """Scatter: mean graph entropy vs fitted alpha, one point per topology."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    topo_entropy = _load_entropy_by_topo(data_roots)

    for topo, entropies in topo_entropy.items():
        if not entropies:
            continue
        col = TOPO_COLORS.get(topo, "#666")
        mrk = TOPO_MARKERS.get(topo, "o")
        lbl = TOPO_LABELS.get(topo, topo)

        # Get event sizes for this topology
        sizes: List[float] = []
        for root in data_roots:
            if not root.exists():
                continue
            events = _load_all_events(root)
            filtered = filter_events(events, topology=topo)
            sizes.extend(extract_delegation_cascades(filtered))

        if len(sizes) < 30:
            continue

        fit = fit_observable(sizes, topo, verbose=False)
        if fit is None:
            continue

        mean_h = float(np.mean(entropies))
        sem_h  = float(np.std(entropies) / max(len(entropies)**0.5, 1))

        ax.errorbar(mean_h, fit.alpha,
                    xerr=sem_h, yerr=fit.sigma_alpha,
                    fmt=mrk, color=col, ms=7, capsize=3,
                    label=lbl, lw=1.2)

    ax.set_xlabel("Mean graph entropy $H(G)$", fontsize=9)
    ax.set_ylabel("Scaling exponent $\\alpha$", fontsize=9)
    ax.set_title("Structural flexibility vs cascade intensity", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # Reference lines for regime interpretation
    ax.axhline(3.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(1.5, color="gray", lw=0.8, ls=":",  alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.98, 3.05, "fragile (α>3)",
            ha="right", fontsize=6, color="gray")
    ax.text(ax.get_xlim()[1] * 0.98, 1.55, "collusion risk (α<1.5)",
            ha="right", fontsize=6, color="gray")

    fig.tight_layout()

    from visualization import _save
    _save(fig, out_dir, figname)
    return fig
