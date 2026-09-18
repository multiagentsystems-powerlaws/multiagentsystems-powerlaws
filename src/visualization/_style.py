"""
visualization/_style.py
------------------------
Shared matplotlib style for all paper figures.
Import this at the top of every visualization module.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# Paper-ready style
STYLE = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "lines.linewidth":   1.5,
    "lines.markersize":  4,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}

# Topology colour palette (consistent across all figures)
TOPO_COLORS = {
    "chain":              "#4C72B0",
    "star":               "#DD8452",
    "tree":               "#55A868",
    "full_mesh":          "#C44E52",
    "sparse_mesh":        "#8172B2",
    "hybrid_modular":     "#937860",
    "dynamic_reputation": "#DA8BC3",
}

TOPO_MARKERS = {
    "chain":              "o",
    "star":               "s",
    "tree":               "^",
    "full_mesh":          "D",
    "sparse_mesh":        "v",
    "hybrid_modular":     "P",
    "dynamic_reputation": "*",
}

TOPO_LABELS = {
    "chain":              "Chain",
    "star":               "Star",
    "tree":               "Tree",
    "full_mesh":          "Full Mesh",
    "sparse_mesh":        "Sparse Mesh",
    "hybrid_modular":     "Hybrid Modular",
    "dynamic_reputation": "Dyn. Reputation",
}

BENCH_COLORS = {
    "MARBLE":    "#2196F3",
    "GAIA":      "#4CAF50",
    "REALM":     "#FF9800",
    "SWE-bench": "#9C27B0",
}

OBSERVABLE_LABELS = {
    "delegation_cascade":  "Delegation cascade size",
    "revision_wave":       "Revision wave size",
    "contradiction_burst": "Contradiction burst size",
    "merge_fanin":         "Merge fan-in",
    "tce_per_run":         "Total cognitive effort (tokens)",
    "influence_per_agent": "Agent influence score",
}


def apply_style():
    """Apply the paper style globally."""
    plt.rcParams.update(STYLE)


def paper_fig(nrows=1, ncols=1, width=7.0, height=None, **kwargs):
    """Create a figure pre-configured for the paper."""
    apply_style()
    if height is None:
        height = width * 0.65 * (nrows / max(ncols, 1))
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
