"""
visualization/appendix/appendix_figures.py
-------------------------------------------
All 6 appendix figures. Each is a parameterised variant of main figures,
sliced by benchmark, topology, or agent scale for robustness claims.

  A1: CCDF per benchmark (one panel per benchmark)
  A2: CCDF per topology  (one panel per topology)
  A3: CCDF per agent scale (one panel per N)
  A4: Event size histograms (log-binned)
  A5: Log-binned distributions with power-law overlay
  A6: Raw vs fitted distributions comparison
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from visualization._style import (
    apply_style, TOPO_COLORS, TOPO_LABELS, BENCH_COLORS, OBSERVABLE_LABELS
)
from event_extraction.coordination import (
    extract_delegation_cascades, filter_events, _load_all_events
)
from tail_fitting.powerlaw_fit import fit_observable


def _ccdf(arr):
    x = np.sort(arr)
    p = 1.0 - np.arange(len(x)) / len(x)
    return x, p

def _save(fig, out_dir, name):
    from visualization import _save as vs
    vs(fig, out_dir, name)


# ── A1: CCDF per benchmark ────────────────────────────────────────

def plot_ccdf_per_benchmark(
    data_roots: List[Path],
    out_dir:    Path,
    observable: str = "delegation_cascade",
    figname:    str = "appendix_a1_ccdf_per_benchmark",
):
    apply_style()
    benches = [r.name for r in data_roots if r.exists()]
    ncols   = min(len(benches), 2)
    nrows   = int(np.ceil(len(benches) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, nrows * 2.8),
                              squeeze=False)
    for idx, root in enumerate(data_roots):
        if not root.exists():
            continue
        ax    = axes[idx // ncols][idx % ncols]
        bench = root.name
        col   = BENCH_COLORS.get(bench, "#666")
        events = _load_all_events(root)
        sizes  = extract_delegation_cascades(events)
        arr    = np.array([s for s in sizes if s > 0])
        ax.set_title(bench, fontsize=8)
        if len(arr) < 20:
            ax.text(0.5,0.5,f"n={len(arr)}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=7)
            continue
        x, p = _ccdf(arr)
        ax.loglog(x, p, ".", color=col, alpha=0.4, ms=2, rasterized=True)
        fit = fit_observable(arr, bench, verbose=False)
        if fit and fit.n_tail >= 20:
            xs = np.logspace(np.log10(fit.x_min), np.log10(arr.max()), 60)
            ys = (xs / fit.x_min) ** (-(fit.alpha - 1))
            xm = np.searchsorted(x, fit.x_min)
            sc = p[xm] if xm < len(p) else 1.0
            ax.loglog(xs, ys*sc, "-", color=col, lw=1.5,
                      label=f"α={fit.alpha:.2f}")
            ax.legend(fontsize=6, frameon=False)
        ax.set_xlabel("Size $x$", fontsize=7)
        ax.set_ylabel("$P(X≥x)$", fontsize=7)
    for idx in range(len(data_roots), nrows*ncols):
        axes[idx//ncols][idx%ncols].set_visible(False)
    fig.suptitle(f"Appendix A1: CCDF per benchmark — "
                 f"{OBSERVABLE_LABELS.get(observable, observable)}", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, figname)
    return fig


# ── A2: CCDF per topology ─────────────────────────────────────────

def plot_ccdf_per_topology(
    data_roots: List[Path],
    out_dir:    Path,
    observable: str = "delegation_cascade",
    figname:    str = "appendix_a2_ccdf_per_topology",
):
    apply_style()
    topos = list(TOPO_COLORS.keys())
    ncols = 4
    nrows = int(np.ceil(len(topos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, nrows * 2.0),
                              squeeze=False)
    for idx, topo in enumerate(topos):
        ax  = axes[idx//ncols][idx%ncols]
        col = TOPO_COLORS[topo]
        sizes = []
        for root in data_roots:
            if not root.exists():
                continue
            events   = _load_all_events(root)
            filtered = filter_events(events, topology=topo)
            sizes.extend(extract_delegation_cascades(filtered))
        arr = np.array([s for s in sizes if s > 0])
        ax.set_title(TOPO_LABELS.get(topo,topo), fontsize=7, pad=2)
        if len(arr) < 20:
            ax.text(0.5,0.5,f"n={len(arr)}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=6)
            continue
        x, p = _ccdf(arr)
        ax.loglog(x, p, ".", color=col, alpha=0.4, ms=2, rasterized=True)
        fit = fit_observable(arr, topo, verbose=False)
        if fit and fit.n_tail >= 20:
            xs = np.logspace(np.log10(fit.x_min), np.log10(arr.max()), 40)
            ys = (xs / fit.x_min) ** (-(fit.alpha - 1))
            xm = np.searchsorted(x, fit.x_min)
            sc = p[xm] if xm < len(p) else 1.0
            ax.loglog(xs, ys*sc, "-", color=col, lw=1.2,
                      label=f"α={fit.alpha:.2f}")
            ax.legend(fontsize=5, frameon=False)
        ax.set_xlabel("Size", fontsize=6)
        ax.set_ylabel("CCDF", fontsize=6)
        ax.tick_params(labelsize=5)
    for idx in range(len(topos), nrows*ncols):
        axes[idx//ncols][idx%ncols].set_visible(False)
    fig.suptitle("Appendix A2: CCDF per topology", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, figname)
    return fig


# ── A3: CCDF per agent scale ──────────────────────────────────────

def plot_ccdf_per_scale(
    data_roots: List[Path],
    out_dir:    Path,
    observable: str = "delegation_cascade",
    figname:    str = "appendix_a3_ccdf_per_scale",
):
    import json
    from collections import defaultdict
    apply_style()
    scale_data: dict = defaultdict(list)

    for root in data_roots:
        if not root.exists():
            continue
        for topo_dir in root.iterdir():
            if not topo_dir.is_dir():
                continue
            for n_dir in topo_dir.iterdir():
                if not n_dir.is_dir() or not n_dir.name.startswith("n"):
                    continue
                try:
                    n = int(n_dir.name[1:])
                except ValueError:
                    continue
                events = []
                for ep in n_dir.rglob("events.jsonl"):
                    with open(ep) as f:
                        for line in f:
                            try:
                                events.append(json.loads(line.strip()))
                            except Exception:
                                pass
                sizes = extract_delegation_cascades(events)
                scale_data[n].extend(sizes)

    scales = sorted(scale_data.keys())
    if not scales:
        print("  A3: no multi-scale data")
        return None

    ncols = min(len(scales), 4)
    nrows = int(np.ceil(len(scales) / ncols))
    cmap  = plt.get_cmap("viridis", len(scales))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, nrows * 2.2),
                              squeeze=False)
    for idx, n in enumerate(scales):
        ax  = axes[idx//ncols][idx%ncols]
        col = cmap(idx / max(len(scales)-1, 1))
        arr = np.array([s for s in scale_data[n] if s > 0])
        ax.set_title(f"N={n}", fontsize=8)
        if len(arr) < 20:
            ax.text(0.5,0.5,f"n={len(arr)}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=7)
            continue
        x, p = _ccdf(arr)
        ax.loglog(x, p, ".", color=col, alpha=0.5, ms=2, rasterized=True)
        fit = fit_observable(arr, f"N={n}", verbose=False)
        if fit and fit.n_tail >= 20:
            xs = np.logspace(np.log10(fit.x_min), np.log10(arr.max()), 40)
            ys = (xs / fit.x_min) ** (-(fit.alpha - 1))
            xm = np.searchsorted(x, fit.x_min)
            sc = p[xm] if xm < len(p) else 1.0
            ax.loglog(xs, ys*sc, "-", color=col, lw=1.5,
                      label=f"α={fit.alpha:.2f}")
            ax.legend(fontsize=6, frameon=False)
        ax.set_xlabel("Size", fontsize=7)
        ax.set_ylabel("CCDF", fontsize=7)
    for idx in range(len(scales), nrows*ncols):
        axes[idx//ncols][idx%ncols].set_visible(False)
    fig.suptitle("Appendix A3: CCDF per agent scale", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, figname)
    return fig


# ── A4: Log-binned histograms ──────────────────────────────────────

def plot_event_histograms(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "appendix_a4_histograms",
):
    apply_style()
    observables = ["delegation_cascade", "merge_fanin", "revision_wave"]
    from event_extraction.coordination import (
        extract_delegation_cascades, extract_merge_fanin, extract_revision_waves
    )
    extractors = {
        "delegation_cascade": extract_delegation_cascades,
        "merge_fanin":        extract_merge_fanin,
        "revision_wave":      extract_revision_waves,
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for ax, obs in zip(axes, observables):
        sizes = []
        for root in data_roots:
            if not root.exists():
                continue
            events = _load_all_events(root)
            sizes.extend(extractors[obs](events))
        arr = np.array([s for s in sizes if s > 0])
        if len(arr) < 5:
            ax.text(0.5, 0.5, "insufficient data", ha="center",
                    va="center", transform=ax.transAxes, fontsize=7)
        else:
            bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 20)
            ax.hist(arr, bins=bins, color="#4C72B0", alpha=0.7, edgecolor="white")
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(OBSERVABLE_LABELS.get(obs, obs), fontsize=7)
        ax.set_xlabel("Size", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
    fig.suptitle("Appendix A4: Event size histograms (log-binned)", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, figname)
    return fig


# ── A5: Log-binned distributions with PL overlay ──────────────────

def plot_logbinned_distributions(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "appendix_a5_logbinned",
):
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    from event_extraction.coordination import extract_delegation_cascades
    for root in data_roots:
        if not root.exists():
            continue
        bench = root.name
        col   = BENCH_COLORS.get(bench, "#666")
        sizes = extract_delegation_cascades(_load_all_events(root))
        arr   = np.array([s for s in sizes if s > 0])
        if len(arr) < 20:
            continue
        bins   = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 15)
        counts, edges = np.histogram(arr, bins=bins)
        widths = np.diff(edges)
        density = counts / (arr.sum() * widths)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.loglog(centres[counts > 0], density[counts > 0], "o-",
                  color=col, ms=4, lw=1.0, label=bench)
    ax.set_xlabel("Event size $x$", fontsize=9)
    ax.set_ylabel("Probability density", fontsize=9)
    ax.set_title("Appendix A5: Log-binned probability density", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, figname)
    return fig


# ── A6: Raw vs fitted ──────────────────────────────────────────────

def plot_raw_vs_fitted(
    data_roots: List[Path],
    out_dir:    Path,
    figname:    str = "appendix_a6_raw_vs_fitted",
):
    apply_style()
    from event_extraction.coordination import extract_delegation_cascades
    all_sizes = []
    for root in data_roots:
        if root.exists():
            all_sizes.extend(extract_delegation_cascades(_load_all_events(root)))
    arr = np.array([s for s in all_sizes if s > 0])
    if len(arr) < 30:
        print("  A6: insufficient data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Raw CCDF
    x, p = _ccdf(arr)
    axes[0].loglog(x, p, ".", color="#4C72B0", alpha=0.4, ms=2, rasterized=True,
                   label="Empirical CCDF")
    fit = fit_observable(arr, "all", verbose=False)
    if fit and fit.n_tail >= 20:
        xs = np.logspace(np.log10(fit.x_min), np.log10(arr.max()), 60)
        ys = (xs / fit.x_min) ** (-(fit.alpha - 1))
        xm = np.searchsorted(x, fit.x_min)
        sc = p[xm] if xm < len(p) else 1.0
        axes[0].loglog(xs, ys*sc, "r-", lw=1.5, label=f"PL α={fit.alpha:.2f}")
        # Lognormal via powerlaw package
        try:
            import powerlaw as pl_pkg
            fo = pl_pkg.Fit(arr, xmin=fit.x_min, verbose=False)
            fo.lognormal.plot_ccdf(ax=axes[0], color="g", ls="--",
                                    lw=1.0, label="Lognormal")
        except Exception:
            pass
        axes[0].axvline(fit.x_min, color="gray", ls=":", lw=0.8, label=f"x_min={fit.x_min:.1f}")
    axes[0].legend(fontsize=6, frameon=False)
    axes[0].set_xlabel("Size $x$", fontsize=8)
    axes[0].set_ylabel("$P(X≥x)$", fontsize=8)
    axes[0].set_title("Empirical vs fitted", fontsize=8)

    # Residuals: log(empirical) - log(fitted)
    if fit and fit.n_tail >= 20:
        tail_mask = arr >= fit.x_min
        tail_x    = np.sort(arr[tail_mask])
        tail_p    = 1.0 - np.arange(tail_mask.sum()) / tail_mask.sum()
        fitted_p  = (tail_x / fit.x_min) ** (-(fit.alpha - 1))
        fitted_p  = fitted_p * (tail_p[0] / fitted_p[0])   # rescale
        residuals = np.log10(np.maximum(tail_p, 1e-10)) - \
                    np.log10(np.maximum(fitted_p, 1e-10))
        axes[1].semilogx(tail_x, residuals, ".", color="#4C72B0",
                          alpha=0.4, ms=2, rasterized=True)
        axes[1].axhline(0, color="red", lw=0.8, ls="--")
        axes[1].set_xlabel("Size $x$", fontsize=8)
        axes[1].set_ylabel("log(empirical/fitted)", fontsize=8)
        axes[1].set_title("Residuals", fontsize=8)

    fig.suptitle("Appendix A6: Raw CCDF vs fitted power law", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, figname)
    return fig


# ── Master caller ──────────────────────────────────────────────────

def plot_all_appendix_figures(data_roots: List[Path], out_dir: Path):
    """Generate all 6 appendix figures."""
    print("\nGenerating appendix figures...")
    for fn, name in [
        (plot_ccdf_per_benchmark,    "A1: CCDF per benchmark"),
        (plot_ccdf_per_topology,     "A2: CCDF per topology"),
        (plot_ccdf_per_scale,        "A3: CCDF per agent scale"),
        (plot_event_histograms,      "A4: Event size histograms"),
        (plot_logbinned_distributions,"A5: Log-binned distributions"),
        (plot_raw_vs_fitted,         "A6: Raw vs fitted"),
    ]:
        print(f"  {name}")
        try:
            fn(data_roots, out_dir)
        except Exception as e:
            import traceback
            print(f"    FAIL: {traceback.format_exc().splitlines()[-1]}")
