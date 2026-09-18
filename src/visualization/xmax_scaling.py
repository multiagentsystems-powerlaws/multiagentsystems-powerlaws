"""
src/visualization/xmax_scaling.py

Scaling law plots: ⟨x_max⟩ vs N for each observable and topology.

Paper Figure 7 (or equivalent):
  - x-axis: system size N (log scale)
  - y-axis: mean empirical maximum ⟨x_max⟩ across seeds (log scale)
  - One curve per topology
  - Power-law fit overlay: ⟨x_max⟩ ~ N^{mu}
  - mu annotation on each curve

This plot tests the EVT prediction: for power-law distributed events
with exponent alpha, the expected maximum scales as:
  ⟨x_max⟩ ~ N^{1/(alpha-1)}

If your empirical mu matches 1/(alpha-1) from the CCDF fits, the
two measurements are internally consistent.

Input:
  For each (topology, N, seed): one FitResult from process_condition().
  xmax_by_condition: Dict[str, float]
    key format: "{topology}_{n_str}_{seed}"   e.g. "chain_n64_s0"
    value: empirical x_max for that run (max(data) for the observable)

Usage:
    from visualization.xmax_scaling import plot_xmax_scaling

    plot_xmax_scaling(
        xmax_data={
            "chain":  {8: [12, 14, 11], 16: [23, 25, 20], 64: [61, 58, 70]},
            "star":   {8: [18, 21, 16], 16: [41, 38, 44], 64: [122, 119, 130]},
        },
        observable_label="TCE",
        out_dir=Path("figures/"),
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Style ─────────────────────────────────────────────────────

TOPOLOGY_COLORS = {
    "chain":             "#4878CF",
    "star":              "#D65F5F",
    "tree":              "#6ACC65",
    "full_mesh":         "#B47CC7",
    "sparse_mesh":       "#C4AD66",
    "hybrid_modular":    "#77BEDB",
    "dynamic_reputation":"#F7A551",
}

TOPOLOGY_MARKERS = {
    "chain":             "o",
    "star":              "s",
    "tree":              "^",
    "full_mesh":         "D",
    "sparse_mesh":       "v",
    "hybrid_modular":    "P",
    "dynamic_reputation":"X",
}


def _apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   7,
        "figure.dpi":        150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "lines.linewidth":   1.2,
    })


# ── Scaling fit ───────────────────────────────────────────────

def fit_scaling_exponent(
    N_vals:    List[int],
    xmax_mean: List[float],
) -> Tuple[float, float]:
    """
    Fit ⟨x_max⟩ ~ N^mu via OLS on log-log scale.
    Returns (mu, intercept_log) where mu is the scaling exponent.
    """
    valid = [(n, x) for n, x in zip(N_vals, xmax_mean) if n > 0 and x > 0]
    if len(valid) < 2:
        return float("nan"), float("nan")
    log_n = np.array([np.log(n) for n, _ in valid])
    log_x = np.array([np.log(x) for _, x in valid])
    mu, intercept = np.polyfit(log_n, log_x, 1)
    return float(mu), float(intercept)


# ── Main plot ─────────────────────────────────────────────────

def plot_xmax_scaling(
    xmax_data:          Dict[str, Dict[int, List[float]]],
    observable_label:   str,
    out_dir:            Path,
    alpha_predictions:  Optional[Dict[str, float]] = None,
    figname:            str = "xmax_scaling",
    figsize:            tuple = (5, 4),
) -> plt.Figure:
    """
    Plot ⟨x_max⟩ vs N for each topology.

    Args:
        xmax_data:
            {topology: {N: [xmax_seed0, xmax_seed1, ...]}}
            Collect x_max = max(data) per observable per run,
            then group by (topology, N).

        observable_label:
            String for axis labels and title, e.g. "TCE".

        out_dir:
            Directory to save figure.

        alpha_predictions:
            Optional {topology: alpha} from CCDF fits.
            If provided, overlays the EVT prediction mu = 1/(alpha-1)
            as a dashed line alongside the empirical fit.

        figname:
            Output filename stem.

        figsize:
            Figure dimensions in inches.

    Returns the matplotlib Figure.
    """
    _apply_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    for topo, n_dict in sorted(xmax_data.items()):
        color  = TOPOLOGY_COLORS.get(topo, "#888")
        marker = TOPOLOGY_MARKERS.get(topo, "o")

        N_vals    = sorted(n_dict.keys())
        xmax_mean = [np.mean(n_dict[n]) for n in N_vals]
        xmax_sem  = [np.std(n_dict[n]) / np.sqrt(len(n_dict[n]))
                     for n in N_vals]

        # Empirical points with error bars
        ax.errorbar(N_vals, xmax_mean, yerr=xmax_sem,
                    fmt=marker, color=color, ms=5, lw=1.2, capsize=2,
                    label=topo)

        # OLS scaling fit
        mu, intercept = fit_scaling_exponent(N_vals, xmax_mean)
        if not np.isnan(mu) and len(N_vals) >= 3:
            N_line = np.logspace(np.log10(min(N_vals)),
                                 np.log10(max(N_vals)), 100)
            x_line = np.exp(intercept) * N_line ** mu
            ax.loglog(N_line, x_line, "-", color=color, lw=0.8, alpha=0.7)
            # Annotate mu near the end of the line
            ax.text(N_line[-1] * 1.05, x_line[-1],
                    f"μ={mu:.2f}", color=color, fontsize=7, va="center")

        # EVT prediction: mu_pred = 1 / (alpha - 1)
        if alpha_predictions and topo in alpha_predictions:
            alpha_fit = alpha_predictions[topo]
            if alpha_fit > 1:
                mu_pred = 1.0 / (alpha_fit - 1.0)
                N_pred  = np.logspace(np.log10(min(N_vals)),
                                      np.log10(max(N_vals)), 100)
                # Anchor the prediction line at the mean of the first N point
                if xmax_mean:
                    anchor = xmax_mean[0] / (min(N_vals) ** mu_pred)
                    x_pred = anchor * N_pred ** mu_pred
                    ax.loglog(N_pred, x_pred, ":", color=color, lw=0.8, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("System size $N$", fontsize=9)
    ax.set_ylabel(f"$\\langle x_{{\\max}} \\rangle$ — {observable_label}", fontsize=9)
    ax.set_title(f"Scaling of empirical maximum: {observable_label}", fontsize=9)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.tick_params(which="both", labelsize=7)
    fig.tight_layout()

    out_path = out_dir / f"{figname}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {out_path}")
    return fig


# ── Helper: collect x_max values from run_pipeline output ────

def collect_xmax_from_runs(
    run_dirs_by_condition: Dict[str, List[Path]],
    observable: str = "tce",
) -> Dict[str, Dict[int, List[float]]]:
    """
    Collect per-run x_max values from run directories.

    Args:
        run_dirs_by_condition: output of run_pipeline.discover_run_dirs()
          keys like "gaia_chain_n64"
        observable: which observable to collect x_max for

    Returns:
        {topology: {N: [xmax_per_seed, ...]}}
    """
    from analysis.run_pipeline import process_run

    result: Dict[str, Dict[int, List[float]]] = {}

    for condition_key, run_dirs in run_dirs_by_condition.items():
        # Parse topology and N from condition key: benchmark_topology_nN
        parts = condition_key.split("_")
        if len(parts) < 3:
            continue
        # topology can have underscores (e.g. full_mesh), N is last part
        n_str = parts[-1]   # "n64"
        if not n_str.startswith("n") or not n_str[1:].isdigit():
            continue
        N = int(n_str[1:])
        topology = "_".join(parts[1:-1])   # everything between benchmark and nN

        if topology not in result:
            result[topology] = {}
        if N not in result[topology]:
            result[topology][N] = []

        for run_dir in run_dirs:
            obs = process_run(run_dir)
            data = obs.get("event_observables", {}).get(observable, [])
            data = [x for x in data if x > 0]
            if data:
                result[topology][N].append(float(max(data)))

    return result