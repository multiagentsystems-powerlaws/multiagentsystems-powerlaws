"""
tables/generate_tables.py
--------------------------
Paper table suite — 5 main tables + 2 appendix tables.

Table 1  Power-law fit statistics for primary graph-derived observables
Table 2  Scaling exponent (α) across topologies and agent scales
Table 3  Model comparison summary: PL vs alternatives
Table 4  Influence concentration and inequality by topology
Table 5  Task quality and efficiency by topology

Appendix A  Heuristic tail interpretation guide
Appendix B  Experiment design configuration

Each table has one clear job:
  T1 → does heavy-tail structure exist?
  T2 → how does α change across settings?
  T3 → does PL actually beat alternatives?
  T4 → do few agents dominate? (the "few players" claim)
  T5 → does concentration connect to real performance?
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _try_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


def _write(rows, pd, out_dir: Path, stem: str, caption: str, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  {stem}: no data")
        return
    if pd:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{stem}.csv", index=False)
        try:
            latex = df.to_latex(
                index=False, float_format="%.3f", escape=False,
                caption=caption, label=label,
            )
            (out_dir / f"{stem}.tex").write_text(latex)
        except Exception:
            pass
        print(f"  {stem} → .csv + .tex  ({len(rows)} rows)")
    else:
        import csv
        with open(out_dir / f"{stem}.tsv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {stem} → .tsv  ({len(rows)} rows)")


def _gini(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n   = len(arr)
    return float((2 * (np.arange(1, n+1) * arr).sum() / (n * arr.sum()) - (n+1)/n))


def _top_share(arr: np.ndarray, pct: float = 0.05) -> float:
    """Fraction of total held by top pct% of agents."""
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return 0.0
    arr   = np.sort(arr)[::-1]
    k     = max(1, int(np.ceil(len(arr) * pct)))
    return float(arr[:k].sum() / arr.sum())


# Human-readable labels for paper tables
OBSERVABLE_LABELS: Dict[str, str] = {
    "agent_unique_descendant_reach": "Agent unique descendant reach",
    "agent_out_strength":            "Agent propagation strength",
    "claim_descendant_count":        "Claim descendant count",
    "cascade_size":                  "Root cascade size",
    "claim_out_degree":              "Claim out-degree",
    "claim_in_degree":               "Claim in-degree",
    "agent_descendant_influence":    "Agent cumulative influence",
    "merge_fan_in":                  "Merge fan-in",
    "claim_depth":                   "Claim depth",
}

# Primary observables for the paper — order matters (strongest first)
PRIMARY_OBSERVABLES = [
    "agent_unique_descendant_reach",
    "agent_out_strength",
    "claim_descendant_count",
    "cascade_size",
    "claim_out_degree",
    "claim_in_degree",
]


# ─────────────────────────────────────────────────────────────────
# Table 1 — Power-law fit statistics (graph-first)
# ─────────────────────────────────────────────────────────────────

def generate_table1(
    data_roots:   List[Path],
    out_dir:      Path,
    analysis_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Power-law fit stats per primary observable × topology × benchmark.
    Uses pre-computed fits_by_group.csv when available.
    Falls back to inline graph extraction + fitting.
    """
    pd = _try_pandas()

    # ── Fast path ────────────────────────────────────────────────
    if analysis_dir and pd:
        fits_path = analysis_dir / "fits_by_group.csv"
        if not fits_path.exists():
            fits_path = analysis_dir / "fit_results.csv"
        if fits_path.exists():
            df = pd.read_csv(fits_path)
            # Normalize column names to lowercase (fits_to_dataframe uses Title Case)
            df.columns = [c.lower() for c in df.columns]
            # Keep only primary observables
            if "observable" in df.columns:
                df = df[df["observable"].isin(PRIMARY_OBSERVABLES)]
            cols_wanted = [
                "observable", "topology", "benchmark", "num_agents",
                "n_tail", "tail_fraction", "x_min",
                "alpha", "sigma_alpha", "ks_statistic",
                "comparison_vs_lognormal", "comparison_vs_exponential",
                "comparison_vs_truncated_pl",
                "best_model_by_aic", "ols_loglog_slope",
                "pl_plausible", "pl_supported", "regime", "quality_tag",
            ]
            available = [c for c in cols_wanted if c in df.columns]
            sort_by = [c for c in ["observable", "topology", "benchmark"] if c in df.columns]
            df_out = df[available].sort_values(sort_by).reset_index(drop=True)                      if sort_by else df[available].reset_index(drop=True)
            _write(df_out.to_dict("records"), pd, out_dir,
                   "table1_pl_fits",
                   "Power-law fit statistics for primary graph-derived observables",
                   "tab:pl_fits")
            return str(out_dir / "table1_pl_fits.tex")

    # ── Inline path: graph extraction + fit ──────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from event_extraction.graph_builder import (
        extract_graph_rows, graph_observables_from_tables,
    )
    from tail_fitting.powerlaw_fit import fit_observable, OBS_SPECS

    claim_nodes, claim_edges, _, _ = extract_graph_rows(data_roots, verbose=False)
    # Group by (topology, benchmark, num_agents) — consistent with fast path
    group_nodes: dict = defaultdict(list)
    group_edges: dict = defaultdict(list)
    for n in claim_nodes:
        key = (n.get("topology","unknown"), n.get("benchmark","unknown"), n.get("num_agents",0))
        group_nodes[key].append(n)
    for e in claim_edges:
        key = (e.get("topology","unknown"), e.get("benchmark","unknown"), e.get("num_agents",0))
        group_edges[key].append(e)

    rows = []
    for (topo, bench, n_agents) in sorted(group_nodes):
        obs = graph_observables_from_tables(group_nodes[(topo, bench, n_agents)],
                                            group_edges.get((topo, bench, n_agents), []))
        for obs_name in PRIMARY_OBSERVABLES:
            sizes = obs.get(obs_name, [])
            if len(sizes) < 30:
                continue
            spec = OBS_SPECS.get(obs_name, {"discrete": True})
            fit  = fit_observable(sizes, obs_name,
                                  discrete=spec.get("discrete", True),
                                  verbose=False, min_n_total=30, min_n_tail=30)
            if fit is None:
                continue
            rows.append({
                "Observable":         obs_name,
                "Topology":           topo,
                "Benchmark":          bench,
                    "N_agents":           n_agents,
                "n_tail":             fit.n_tail,
                "tail_fraction":      fit.tail_fraction,
                "x_min":              round(fit.x_min, 2),
                "α":                  round(fit.alpha, 3),
                "σ_α":                round(fit.sigma_alpha, 3),
                "KS_stat":            round(fit.ks_statistic, 3),
                "vs_lognormal":       fit.comparison_vs_lognormal,
                "vs_exponential":     fit.comparison_vs_exponential,
                "vs_truncated_pl":    fit.comparison_vs_truncated_pl,
                "best_model_AIC":     fit.best_model_by_aic,
                "OLS_slope":          round(fit.ols_loglog_slope, 3),
                "PL_plausible":       "✓" if fit.pl_plausible else "✗",
                "PL_supported":       "✓" if fit.pl_supported else "✗",
                "Regime":             fit.regime,
                "Quality":            fit.quality_tag,
            })

    _write(rows, pd, out_dir, "table1_pl_fits",
           "Power-law fit statistics for primary graph-derived observables",
           "tab:pl_fits")
    return str(out_dir / "table1_pl_fits.tex")


# ─────────────────────────────────────────────────────────────────
# Table 2 — Scaling exponent α across topologies and scales
# ─────────────────────────────────────────────────────────────────

def generate_table2(
    data_roots:   List[Path],
    out_dir:      Path,
    analysis_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    α values per (observable, topology) stratified by agent scale N.
    Answers: is α stable, topology-specific, or scale-dependent?
    """
    pd = _try_pandas()
    if pd is None:
        print("  Table 2: pandas required")
        return None

    # Load grouped fits
    fits_df = None
    if analysis_dir:
        for fname in ["fits_by_group.csv", "fit_results.csv"]:
            p = analysis_dir / fname
            if p.exists():
                fits_df = pd.read_csv(p)
                break

    if fits_df is None:
        print("  Table 2: no grouped fit results found — run analyse stage first")
        return None

    if "observable" not in fits_df.columns:
        print("  Table 2: fit results missing 'observable' column")
        return None

    df = fits_df[fits_df["observable"].isin(PRIMARY_OBSERVABLES)].copy()
    if df.empty:
        print("  Table 2: no primary observable fits found")
        return None

    # Determine available scale column
    scale_col = None
    for c in ["num_agents", "n_agents", "scale"]:
        if c in df.columns:
            scale_col = c
            break

    rows = []
    for obs in PRIMARY_OBSERVABLES:
        obs_df = df[df["observable"] == obs]
        for topo in sorted(obs_df["topology"].unique() if "topology" in obs_df.columns else []):
            topo_df = obs_df[obs_df["topology"] == topo] if "topology" in obs_df.columns else obs_df
            row: dict = {"Observable": obs, "Topology": topo}

            # Pooled α
            alpha_col = "alpha" if "alpha" in topo_df.columns else "α"
            if alpha_col in topo_df.columns:
                all_alpha = topo_df[alpha_col].dropna()
                row["α_pooled"]     = round(float(all_alpha.mean()), 3) if len(all_alpha) else None
                row["σ_pooled"]     = round(float(all_alpha.std()), 3)  if len(all_alpha) > 1 else None
                ps = topo_df["pl_supported"] if "pl_supported" in topo_df.columns else pd.Series(dtype=object)
                ps_norm = ps.astype(str).str.lower().isin(["true", "1", "yes", "✓"])
                row["n_supported"] = int(ps_norm.sum())

            # Per-scale α columns
            if scale_col:
                for n in sorted(topo_df[scale_col].dropna().unique()):
                    n_int = int(n)
                    n_df  = topo_df[topo_df[scale_col] == n]
                    a_vals = n_df[alpha_col].dropna() if alpha_col in n_df.columns else pd.Series()
                    row[f"α@N={n_int}"] = round(float(a_vals.mean()), 3) if len(a_vals) else None

            rows.append(row)

    if not rows:
        # Fallback: just show pooled per observable
        for obs in PRIMARY_OBSERVABLES:
            obs_df = df[df["observable"] == obs]
            alpha_col = "alpha" if "alpha" in obs_df.columns else "α"
            if alpha_col in obs_df.columns:
                a = obs_df[alpha_col].dropna()
                rows.append({
                    "Observable": obs,
                    "α_mean": round(float(a.mean()), 3) if len(a) else None,
                    "α_std":  round(float(a.std()), 3)  if len(a) > 1 else None,
                    "n_fits": len(a),
                })

    _write(rows, pd, out_dir, "table2_alpha_across_settings",
           r"Scaling exponent $\alpha$ across topologies and agent scales",
           "tab:alpha_settings")
    return str(out_dir / "table2_alpha_across_settings.tex")


# ─────────────────────────────────────────────────────────────────
# Table 3 — Model comparison summary (PL vs alternatives)
# ─────────────────────────────────────────────────────────────────

def generate_table3(
    data_roots:   List[Path],
    out_dir:      Path,
    analysis_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    For each primary observable: % groups where PL beats alternatives.
    Answers: are we really seeing power laws, or just noisy alternatives?
    Crucial for reviewer trust.
    """
    pd = _try_pandas()
    if pd is None:
        print("  Table 3: pandas required")
        return None

    fits_df = None
    if analysis_dir:
        for fname in ["fits_by_group.csv", "fit_results.csv"]:
            p = analysis_dir / fname
            if p.exists():
                fits_df = pd.read_csv(p)
                break

    if fits_df is None:
        print("  Table 3: no grouped fit results found")
        return None

    if "observable" not in fits_df.columns:
        print("  Table 3: fit results missing 'observable' column")
        return None
    df = fits_df[fits_df["observable"].isin(PRIMARY_OBSERVABLES)].copy()

    if df.empty:
        print("  Table 3: no primary observable fits found")
        return None

    def _pct(col_name: str, value) -> "Optional[float]":
        if col_name not in grp.columns:
            return None
        series = grp[col_name]
        if len(series) == 0:
            return None
        return round(100.0 * (series == value).sum() / len(series), 1)

    rows = []
    obs_col = "observable" if "observable" in df.columns else None
    groups  = df.groupby(obs_col) if obs_col else [("all", df)]

    for obs_name, grp in groups:
        if obs_name not in PRIMARY_OBSERVABLES:
            continue
        n = len(grp)
        row = {
            "Observable":           obs_name,
            "n_groups":             n,
            "% PL > lognormal":     _pct("comparison_vs_lognormal", "pl_better"),
            "% PL > exponential":   _pct("comparison_vs_exponential", "pl_better"),
            "% PL > truncated_pl":  _pct("comparison_vs_truncated_pl", "pl_better"),
            "% best_AIC = PL":      _pct("best_model_by_aic", "power_law"),
            "% PL_plausible":       _pct("pl_plausible", True),
            "% PL_supported":       _pct("pl_supported", True),
        }
        rows.append(row)

    _write(rows, pd, out_dir, "table3_model_comparison",
           "Power-law vs alternative model comparison across fitting groups",
           "tab:model_comparison")
    return str(out_dir / "table3_model_comparison.tex")


# ─────────────────────────────────────────────────────────────────
# Table 4 — Influence concentration and inequality by topology
# ─────────────────────────────────────────────────────────────────

def generate_table4(
    data_roots:    List[Path],
    out_dir:       Path,
    extracted_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Inequality statistics per topology for primary influence observables.
    The 'few players' table: Gini, top-5% share, p90, p99, xmax.
    Most important for the paper's conceptual hook.
    """
    pd = _try_pandas()

    # ── Load graph observables ────────────────────────────────────
    # Try claim_nodes.csv + claim_edges.csv first (fastest)
    obs_by_topo: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    if extracted_dir and (extracted_dir / "claim_nodes.csv").exists() and pd:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            from event_extraction.graph_builder import graph_observables_from_tables
            cn = pd.read_csv(extracted_dir / "claim_nodes.csv").to_dict("records")
            ce = pd.read_csv(extracted_dir / "claim_edges.csv").to_dict("records") \
                 if (extracted_dir / "claim_edges.csv").exists() else []
            # Split by topology
            topo_nodes: dict = defaultdict(list)
            topo_edges: dict = defaultdict(list)
            for n in cn:
                topo_nodes[n.get("topology", "unknown")].append(n)
            for e in ce:
                topo_edges[e.get("topology", "unknown")].append(e)
            for topo in topo_nodes:
                g_obs = graph_observables_from_tables(topo_nodes[topo], topo_edges[topo])
                for obs_name, vals in g_obs.items():
                    obs_by_topo[topo][obs_name].extend(vals)
        except Exception as e:
            print(f"  Table 4: claim CSV load failed ({e}), falling back to raw graph extraction")

    # Fallback: re-extract from raw logs (graph-derived only — no event-native fallback)
    if not obs_by_topo:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from event_extraction.graph_builder import extract_graph_rows, graph_observables_from_tables
        cn, ce, _, _ = extract_graph_rows(data_roots, verbose=False)
        topo_nodes = defaultdict(list)
        topo_edges = defaultdict(list)
        for n in cn:
            topo_nodes[n.get("topology", "unknown")].append(n)
        for e in ce:
            topo_edges[e.get("topology", "unknown")].append(e)
        for topo in topo_nodes:
            g_obs = graph_observables_from_tables(topo_nodes[topo], topo_edges[topo])
            for obs_name, vals in g_obs.items():
                obs_by_topo[topo][obs_name].extend(vals)

    if not obs_by_topo:
        print("  Table 4: no graph observable data found")
        return None

    INEQ_OBSERVABLES = [
        "agent_unique_descendant_reach",
        "agent_out_strength",
        "claim_descendant_count",
        "cascade_size",
    ]

    rows = []
    for topo in sorted(obs_by_topo):
        for obs_name in INEQ_OBSERVABLES:
            vals = [v for v in obs_by_topo[topo].get(obs_name, []) if v > 0]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            rows.append({
                "Observable":    obs_name,
                "Topology":      topo,
                "n":             len(arr),
                "mean":          round(float(arr.mean()), 3),
                "median":        round(float(np.median(arr)), 3),
                "x_max":         round(float(arr.max()), 1),
                "Gini":          round(_gini(arr), 4),
                "p90":           round(float(np.percentile(arr, 90)), 3),
                "p99":           round(float(np.percentile(arr, 99)), 3),
                "top-5% share":  round(_top_share(arr, 0.05), 4),
                "top-1% share":  round(_top_share(arr, 0.01), 4),
            })

    _write(rows, pd, out_dir, "table4_influence_concentration",
           "Influence concentration and inequality by topology",
           "tab:concentration")
    return str(out_dir / "table4_influence_concentration.tex")


# ─────────────────────────────────────────────────────────────────
# Table 5 — Task quality and efficiency by topology
# ─────────────────────────────────────────────────────────────────

def generate_table5(
    data_roots:    List[Path],
    out_dir:       Path,
    extracted_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    H2 quality/efficiency metrics per topology.
    Connects structural concentration to real MAS performance.
    """
    pd = _try_pandas()
    if pd is None:
        print("  Table 5: pandas required")
        return None

    run_metrics_path = (extracted_dir / "run_metrics.csv") if extracted_dir else None
    df = None

    if run_metrics_path and run_metrics_path.exists():
        df = pd.read_csv(run_metrics_path)
    else:
        # Load inline from run_metadata.json
        H2_FIELDS = [
            "task_score", "completion_ratio", "coherence_score", "integration_score",
            "tokens_total", "wall_time_seconds", "quality_adjusted_efficiency",
            "claim_participation_rate", "resolution_rate",
            "revisions_per_claim", "merges_per_claim",
        ]
        rows_raw = []
        for root in data_roots:
            if not root.exists():
                continue
            for meta_path in root.rglob("run_metadata.json"):
                try:
                    meta  = json.loads(meta_path.read_text())
                    extra = meta.get("extra", {})
                    run_id = meta.get("run_id", "")
                    parts  = run_id.split("__")
                    row = {
                        "topology":  parts[1] if len(parts) > 1 else "unknown",
                        "benchmark": parts[0] if len(parts) > 0 else "unknown",
                    }
                    for f in H2_FIELDS:
                        v = meta.get(f)
                        if v is None:
                            v = extra.get(f)
                        row[f] = float(v) if v is not None else None
                    rows_raw.append(row)
                except Exception:
                    pass
        if not rows_raw:
            print("  Table 5: no run metadata found")
            return None
        df = pd.DataFrame(rows_raw)

    if df is None or df.empty:
        print("  Table 5: empty data")
        return None

    QUAL_COLS = [c for c in [
        "task_score", "completion_ratio", "coherence_score", "integration_score",
        "tokens_total", "wall_time_seconds", "quality_adjusted_efficiency",
        "claim_participation_rate", "resolution_rate",
    ] if c in df.columns]

    if "topology" not in df.columns or not QUAL_COLS:
        print("  Table 5: missing topology or quality columns")
        return None

    summary = df.groupby("topology")[QUAL_COLS].agg(["mean", "std"]).round(4)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()

    _write(summary.to_dict("records"), pd, out_dir,
           "table5_quality_efficiency",
           "Task quality and efficiency metrics by topology",
           "tab:quality_efficiency")
    return str(out_dir / "table5_quality_efficiency.tex")


# ─────────────────────────────────────────────────────────────────
# Appendix A — Heuristic tail interpretation guide
# ─────────────────────────────────────────────────────────────────

def generate_appendix_a(out_dir: Path) -> str:
    pd = _try_pandas()
    rows = [
        {"α range": "α < 1.5",
         "Heuristic label": "Collusion risk",
         "Meaning": "A few agents command almost all downstream influence. Extreme events dominate.",
         "Interpretation": "Inspect top-k agents for echo chambers or coordinated bias."},
        {"α range": "1.5 ≤ α < 2.0",
         "Heuristic label": "Strong collective intelligence",
         "Meaning": "Very heavy tails; rare mega-cascades drive most coordination.",
         "Interpretation": "Creative but unpredictable; monitor cascade size distribution."},
        {"α range": "2.0 ≤ α ≤ 3.0",
         "Heuristic label": "Healthy collective intelligence",
         "Meaning": "Moderate heavy tail; robust scale-free coordination.",
         "Interpretation": "Optimal regime for most tasks; consistent with social power laws."},
        {"α range": "3.0 < α ≤ 4.0",
         "Heuristic label": "Fragile reasoning",
         "Meaning": "Light tail; large cascades rare; agents work mostly in isolation.",
         "Interpretation": "Consider richer topology or more cross-agent communication."},
        {"α range": "α > 4.0",
         "Heuristic label": "Near-independent agents",
         "Meaning": "Effectively exponential; no scale-free structure.",
         "Interpretation": "Coordination overhead likely outweighs benefit."},
    ]
    _write(rows, pd, out_dir, "appendix_a_tail_interpretation",
           "Heuristic interpretation guide for observed tail exponents (not formal thresholds)",
           "tab:tail_guide")
    return str(out_dir / "appendix_a_tail_interpretation.tex")


# ─────────────────────────────────────────────────────────────────
# Appendix B — Experiment design
# ─────────────────────────────────────────────────────────────────

def generate_appendix_b(data_roots: List[Path], out_dir: Path) -> str:
    pd = _try_pandas()
    benchmarks: set   = set()
    topologies: set   = set()
    agent_scales: set = set()
    seeds: set        = set()
    n_runs            = 0
    tasks_per_bench: dict = defaultdict(set)

    for root in data_roots:
        if not root.exists():
            continue
        for cfg_path in root.rglob("run_config.json"):
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue
            benchmarks.add(cfg.get("benchmark", "?"))
            topologies.add(cfg.get("topology", "?"))
            agent_scales.add(cfg.get("num_agents", 0))
            seeds.add(cfg.get("seed", 0))
            tasks_per_bench[cfg.get("benchmark", "?")].add(cfg.get("task_id", "?"))
            n_runs += 1

    # Fallback: scan run_metadata.json
    if n_runs == 0:
        for root in data_roots:
            if not root.exists():
                continue
            for meta_path in root.rglob("run_metadata.json"):
                try:
                    meta   = json.loads(meta_path.read_text())
                    run_id = meta.get("run_id", "")
                    parts  = run_id.split("__")
                    if len(parts) >= 5:
                        benchmarks.add(parts[0])
                        topologies.add(parts[1])
                        n_str = parts[2].replace("n", "")
                        if n_str.isdigit():
                            agent_scales.add(int(n_str))
                        seeds.add(parts[3].lstrip("s"))
                        tasks_per_bench[parts[0]].add(parts[4])
                    n_runs += 1
                except Exception:
                    pass

    tasks_str = "; ".join(f"{b}: {len(t)}" for b, t in sorted(tasks_per_bench.items()))
    rows = [
        {"Item": "Benchmarks",          "Value": ", ".join(sorted(benchmarks)) or "—"},
        {"Item": "Tasks per benchmark",  "Value": tasks_str or "—"},
        {"Item": "Topologies",           "Value": ", ".join(sorted(topologies)) or "—"},
        {"Item": "Agent scales (N)",     "Value": ", ".join(str(n) for n in sorted(agent_scales)) or "—"},
        {"Item": "Seeds",                "Value": ", ".join(str(s) for s in sorted(seeds)) or "—"},
        {"Item": "Total runs",           "Value": str(n_runs)},
        {"Item": "Primary observables",  "Value": ", ".join(PRIMARY_OBSERVABLES)},
    ]
    _write(rows, pd, out_dir, "appendix_b_experiment_design",
           "Experimental configuration", "tab:experiment_design")
    return str(out_dir / "appendix_b_experiment_design.tex")


# ─────────────────────────────────────────────────────────────────
# Master generator
# ─────────────────────────────────────────────────────────────────

def generate_all_tables(
    data_roots:    List[Path],
    out_dir:       Path,
    analysis_dir:  Optional[Path] = None,
    extracted_dir: Optional[Path] = None,
):
    """
    Generate the full paper table suite.

    Table 1  Power-law fit statistics (graph-first, primary observables)
    Table 2  α across topologies × scales
    Table 3  Model comparison: PL vs alternatives
    Table 4  Influence concentration / inequality  ← most important
    Table 5  Task quality / efficiency by topology

    Appendix A  Heuristic tail interpretation guide
    Appendix B  Experiment design
    """
    print("\nGenerating tables...")

    generate_table1(data_roots, out_dir, analysis_dir=analysis_dir)
    generate_table2(data_roots, out_dir, analysis_dir=analysis_dir)
    generate_table3(data_roots, out_dir, analysis_dir=analysis_dir)
    generate_table4(data_roots, out_dir, extracted_dir=extracted_dir)
    generate_table5(data_roots, out_dir, extracted_dir=extracted_dir)
    generate_appendix_a(out_dir)
    generate_appendix_b(data_roots, out_dir)

    print(f"\nAll tables → {out_dir}/")