"""
src/tail_fitting/powerlaw_fit.py

Clauset-Newman-Shalizi-correct discrete power-law fitting for coordination
event sizes, using a truncated power law (TPL) as the primary model.

Methodological notes:

  1. All five observables are INTEGER-VALUED. discrete=True enforced throughout.
     The discrete MLE applies the correction x_i / (x_min - 0.5) vs the
     continuous version. Omitting this overestimates alpha by 0.05-0.15.

  2. x_min selected by KS minimisation (Clauset procedure, via the `powerlaw`
     package). Pass xmin= to force a value for sensitivity checks only.

  3. PRIMARY MODEL IS THE TRUNCATED POWER LAW (TPL), P(X=x) ~ x^-alpha * e^(-x/x_c)
     -- not the pure/untruncated power law. alpha and x_c both come from
     fit.truncated_power_law, so every reported alpha carries its matching
     cutoff scale x_c rather than assuming an unbounded tail.

  4. TPL-vs-PL is a NESTED comparison: PL is TPL's boundary case (x_c -> inf,
     i.e. the rate parameter Lambda=1/x_c -> 0). Because the null hypothesis
     sits on the boundary of TPL's parameter space, the standard chi-squared(1)
     LRT is invalid; the correct asymptotic null distribution for a one-sided
     boundary test is a 50:50 mixture of a point mass at 0 and chi-squared(1)
     (Chernoff 1954; Self & Liang 1987). This boundary-aware nested LRT is
     NOT the same test the underlying `powerlaw` package's generic
     distribution_compare() runs. Implemented here as boundary_aware_lrt().

  5. TPL-vs-LN uses the package's non-nested Vuong test (normalized_ratio=True)
     between "truncated_power_law" and "lognormal" (R > 0 favours TPL).

  6. sigma_alpha (fast path) uses the analytic formula (alpha-1)/sqrt(n_tail)
     from Clauset et al. Eq. 4. For a tighter, cluster-aware 95% CI, use
     RUN-CLUSTERED BOOTSTRAP resampling instead (resampling whole runs, not
     individual events, to respect within-run clustering) -- pass run_ids=
     and n_bootstrap>0 to get this; each refit costs ~1-3s so this is
     opt-in, not the default, to keep routine/smoke-test fitting fast. See
     bootstrap_alpha_ci().

  7. GOF (goodness-of-fit) for the fitted TPL: a semiparametric bootstrap
     p-value (simulate synthetic data from the fitted model, refit, compare
     KS distances -- Clauset et al. Sec 5) is the statistically rigorous
     option, opt-in via n_gof_sim>0 (see semiparametric_gof()); the
     fast-path default falls back to a KS<0.1 heuristic (documented as
     approximate, not a real p-value). WARNING: the underlying `powerlaw`
     package's truncated_power_law.generate_random() scales very poorly
     with tail size -- measured ~1.5s at n_tail=100 but ~16s at n_tail=1000
     (roughly linear, not the sub-second cost of the fit itself) -- so at
     tail sizes in the tens of thousands, a single n_gof_sim=100 run could
     take hours. Only enable this for small conditions or with a
     long-running batch job, not interactively.

  8. Holm correction: apply holm_correction() to the TPL-vs-LN and TPL-vs-PL
     p-values across the set of observables being reported together, e.g.
     in fit_all(), to control the family-wise error rate.

  9. Gini computed on TAIL DATA ONLY (arr >= x_min), consistent with alpha,
     KS, and LR tests.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import powerlaw  # type: ignore  (Alstott et al. 2014)
from scipy.stats import chi2  # type: ignore


# ── Observable metadata ───────────────────────────────────────

OBSERVABLES: Dict[str, dict] = {
    "delegation_sizes":     {"label": "Delegation cascade size",  "discrete": True},
    "revision_waves":       {"label": "Revision wave size",       "discrete": True},
    "contradiction_bursts": {"label": "Contradiction burst size",  "discrete": True},
    "merge_fan_in":         {"label": "Merge fan-in",             "discrete": True},
    "tce":                  {"label": "TCE",                      "discrete": True},
}


# ── Result dataclass ──────────────────────────────────────────

@dataclass
class FitResult:
    """All statistics from one truncated-power-law fit. One instance = one table row."""

    observable:    str
    label:         str

    # Sample statistics
    n_total:       int
    n_tail:        int       # samples at or above x_min
    tail_fraction: float     # n_tail / n_total
    x_min:         float
    x_max:         float
    mean:          float
    median:        float
    dynamic_range: int       # distinct integer values in tail

    # Truncated power-law fit: P(X=x) ~ x^-alpha * e^(-x/x_c)
    alpha:         float     # scaling exponent (discrete MLE, TPL)
    x_c:           float     # exponential cutoff scale (TPL), inf if unconstrained
    sigma_alpha:   float     # CI half-width proxy; see ci_method
    alpha_ci_low:  float
    alpha_ci_high: float
    ci_method:     str       # "analytic" (fast path) or "run_clustered_bootstrap"

    ks_stat:       float     # KS distance between empirical tail and fitted TPL
    gof_pvalue:    float     # goodness-of-fit p-value for the fitted TPL
    gof_method:    str       # "ks_heuristic" (fast path) or "semiparametric_bootstrap"

    # Non-nested Vuong test: TPL vs log-normal (R_TPL/LN, R > 0 favours TPL)
    lr_tpl_vs_ln:      float
    lr_tpl_vs_ln_p:    float
    # Non-nested Vuong test: TPL vs exponential
    lr_tpl_vs_exp:     float
    lr_tpl_vs_exp_p:   float
    # Boundary-aware NESTED LRT: TPL vs PL (2*Delta-loglik)
    lr_tpl_vs_pl:      float   # 2 * (loglik_TPL - loglik_PL)
    lr_tpl_vs_pl_p:    float   # boundary-aware (50:50 chi-bar-squared) p-value

    # Verdicts
    tpl_plausible:        bool   # GOF test does not reject TPL (gof_pvalue > 0.1)
    tpl_beats_lognormal:  bool   # lr_tpl_vs_ln > 0 and p < 0.05
    tpl_beats_exponential: bool  # lr_tpl_vs_exp > 0 and p < 0.05
    tpl_beats_pl:          bool  # lr_tpl_vs_pl > 0 and p < 0.05 (truncation is real)

    # Regime classification
    regime: str   # collusion_risk | collective_intelligence | fragile_reasoning

    # Inequality — computed on TAIL DATA ONLY, consistent with other metrics
    gini: float

    tags: List[str] = field(default_factory=list)

    # ── Backward-compat aliases (previous pure-PL-era field names) ────────
    @property
    def pl_plausible(self) -> bool:
        return self.tpl_plausible

    @property
    def pl_beats_lognormal(self) -> bool:
        return self.tpl_beats_lognormal

    @property
    def pl_beats_exponential(self) -> bool:
        return self.tpl_beats_exponential

    @property
    def lr_vs_lognormal(self) -> float:
        return self.lr_tpl_vs_ln

    @property
    def lr_vs_lognormal_p(self) -> float:
        return self.lr_tpl_vs_ln_p

    @property
    def ks_pvalue(self) -> float:
        return self.gof_pvalue

    @property
    def lr_vs_exponential(self) -> float:
        return self.lr_tpl_vs_exp

    @property
    def lr_vs_exponential_p(self) -> float:
        return self.lr_tpl_vs_exp_p

    @property
    def lr_vs_truncated_pl(self) -> float:
        return self.lr_tpl_vs_pl

    @property
    def lr_vs_truncated_pl_p(self) -> float:
        return self.lr_tpl_vs_pl_p


# Alias: some callers use the name OBS_SPECS for this same metadata dict.
OBS_SPECS = OBSERVABLES


# ── Helpers ───────────────────────────────────────────────────

def _regime(alpha: float) -> str:
    if alpha < 1.5:
        return "collusion_risk"
    if alpha <= 3.0:
        return "collective_intelligence"
    return "fragile_reasoning"


def _gini(data: np.ndarray) -> float:
    """Gini coefficient. Input must already be filtered to tail."""
    if len(data) == 0 or data.sum() == 0:
        return 0.0
    data = np.sort(data.astype(float))
    n    = len(data)
    idx  = np.arange(1, n + 1)
    return float((2 * (idx * data).sum() / (n * data.sum())) - (n + 1) / n)


def _ks_distance(dist) -> float:
    """
    KS distance for a fitted powerlaw distribution object.

    Not dist.KS() -- that method is broken in this version of the `powerlaw`
    package (calls compute_distance_metrics(data) without `self.`, always
    raising NameError; its own docstring says it was renamed
    compute_distance_metrics()). Call that directly instead and read .D.
    """
    try:
        dist.compute_distance_metrics()
        return float(dist.D)
    except Exception:
        return float("nan")


def _x_c(tpl) -> float:
    """Cutoff scale x_c = 1/Lambda from a fitted truncated_power_law object."""
    lam = getattr(tpl, "Lambda", 0.0) or 0.0
    return float(1.0 / lam) if lam > 0 else float("inf")


def boundary_aware_lrt(fit) -> tuple[float, float]:
    """
    Boundary-aware nested LRT: TPL vs PL.

    PL is TPL's boundary case (x_c -> inf / Lambda -> 0), so the null
    distribution of 2*Delta-loglik is a 50:50 mixture of a point mass at 0
    and chi-squared(1) (Chernoff 1954), not a plain chi-squared(1).

    Returns (two_delta_ll, boundary_aware_p).
    """
    try:
        ll_tpl = float(fit.truncated_power_law.loglikelihoods(fit.data).sum())
        ll_pl  = float(fit.power_law.loglikelihoods(fit.data).sum())
    except Exception:
        return float("nan"), float("nan")
    two_dll = 2.0 * (ll_tpl - ll_pl)
    if not np.isfinite(two_dll) or two_dll <= 0:
        return (two_dll if np.isfinite(two_dll) else float("nan")), 1.0
    p = 0.5 * (1.0 - chi2.cdf(two_dll, df=1))
    return two_dll, float(p)


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """
    Holm (1979) step-down correction for a family of p-values, preserving
    input order. NaNs pass through unadjusted (excluded from the family).
    """
    indexed = [(p, i) for i, p in enumerate(p_values) if np.isfinite(p)]
    indexed.sort(key=lambda t: t[0])
    m = len(indexed)
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, (p, orig_i) in enumerate(indexed):
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        adjusted[orig_i] = min(running_max, 1.0)
    return adjusted


def bootstrap_alpha_ci(
    data: np.ndarray,
    run_ids: Sequence[str],
    xmin: float,
    discrete: bool,
    n_bootstrap: int = 50,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Run-clustered bootstrap 95% CI for alpha. Resamples whole runs with
    replacement (not individual
    events), pools their data, and refits TPL at the fixed xmin each time --
    respecting the fact that events within a run are not independent samples.
    """
    rng = np.random.default_rng(seed)
    by_run: Dict[str, List[float]] = {}
    for x, rid in zip(data, run_ids):
        by_run.setdefault(rid, []).append(x)
    unique_runs = list(by_run.keys())
    if len(unique_runs) < 2:
        return float("nan"), float("nan")

    alphas = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_runs, size=len(unique_runs), replace=True)
        sample = np.concatenate([np.array(by_run[r]) for r in chosen])
        sample = sample[sample >= xmin]
        if len(sample) < 10:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot_fit = powerlaw.Fit(sample, xmin=xmin, discrete=discrete, verbose=False)
            alphas.append(float(boot_fit.truncated_power_law.alpha))
        except Exception:
            continue

    if len(alphas) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(alphas, [2.5, 97.5])
    return float(lo), float(hi)


def semiparametric_gof(
    fit,
    n_sim: int = 100,
    seed: int = 0,
) -> float:
    """
    Semiparametric bootstrap GOF test for the fitted TPL (Clauset et al. 2009
    Sec. 5): generate n_sim synthetic tails from the
    fitted model, refit each, and take the fraction of synthetic KS distances
    >= the observed KS distance as the p-value. A high p-value means the
    fitted TPL is not rejected as implausible.
    """
    tpl = fit.truncated_power_law
    observed_ks = _ks_distance(tpl)
    if np.isnan(observed_ks):
        return float("nan")

    rng = np.random.default_rng(seed)
    n_tail = int((fit.data >= fit.xmin).sum()) if hasattr(fit, "data") else len(fit.data)
    exceed = 0
    valid  = 0
    for i in range(n_sim):
        try:
            np.random.seed(int(rng.integers(0, 2**31 - 1)))  # generate_random() has no seed= kwarg
            synth = tpl.generate_random(n_tail)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                synth_fit = powerlaw.Fit(synth, xmin=fit.xmin, discrete=fit.discrete, verbose=False)
            synth_ks = _ks_distance(synth_fit.truncated_power_law)
        except Exception:
            continue
        valid += 1
        if synth_ks >= observed_ks:
            exceed += 1

    if valid < n_sim // 2:
        return float("nan")
    return exceed / valid


# ── Core fitter ───────────────────────────────────────────────

def fit_observable(
    data:        List[int],
    name:        str,
    xmin:        Optional[float] = None,
    verbose:     bool = True,
    run_ids:     Optional[List[str]] = None,
    n_bootstrap: int = 50,
    n_gof_sim:   int = 0,
    bootstrap_seed: int = 0,
    discrete:       Optional[bool] = None,
    specs:          Optional[Dict[str, dict]] = None,
    min_n_total:    int = 50,
    min_n_tail:     int = 50,
) -> Optional[FitResult]:
    """
    Fit a discrete truncated power law to a list of positive integer event sizes.

    data         Pooled list across seeds/runs for one condition. Pool first.
    name         Key from OBSERVABLES dict or any label string.
    xmin         Force lower cutoff (None = auto via KS minimisation).
    run_ids      Parallel array to `data` giving the run each sample came
                 from. If provided and n_bootstrap>0, alpha's 95% CI is
                 computed via run-clustered bootstrap; otherwise falls back
                 to the fast analytic sigma formula.
    n_bootstrap  Number of bootstrap refits for the CI (statistically rigorous;
                 each refit costs ~1-3s, so this is meaningful cost for large
                 tails — set to 0 to skip and always use the fast analytic CI).
    n_gof_sim    Number of semiparametric GOF simulations (0 = skip, use the
                 fast KS<0.1 heuristic instead; each simulation costs ~1-3s).
    discrete     Override the discreteness flag looked up from `specs`/OBSERVABLES.
    specs        Override the OBSERVABLES metadata dict for this call.
    min_n_total  Minimum total sample size (relax for smoke tests).
    min_n_tail   Minimum tail sample size after x_min selection.

    Returns None if n_total < min_n_total or n_tail < min_n_tail after x_min selection.
    """
    pairs = [(x, r) for x, r in zip(data, (run_ids or [None] * len(data))) if x and x > 0]
    if not pairs:
        if verbose:
            print(f"  {name}: n=0 — skipping.")
        return None
    arr = np.array([p[0] for p in pairs], dtype=float)
    ids = [p[1] for p in pairs]

    if len(arr) < min_n_total:
        if verbose:
            print(f"  {name}: n={len(arr)} < {min_n_total} — skipping.")
        return None

    meta        = (specs or OBSERVABLES).get(name, {"label": name, "discrete": True})
    is_discrete = discrete if discrete is not None else meta["discrete"]
    label       = meta["label"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(arr, xmin=xmin, discrete=is_discrete, verbose=False)

    tail_mask = arr >= fit.xmin
    tail_data = arr[tail_mask]
    n_tail    = int(tail_mask.sum())

    if n_tail < min_n_tail:
        if verbose:
            print(f"  {name}: n_tail={n_tail} < {min_n_tail} above xmin={fit.xmin:.1f} — skipping.")
        return None

    dynamic_range = int(len(np.unique(tail_data.astype(int))))
    if dynamic_range < 20 and verbose:
        print(f"  [warn] {name}: dynamic_range={dynamic_range} < 20 — "
              f"fit may be unreliable (insufficient distinct tail values).")

    tpl = fit.truncated_power_law
    alpha = float(tpl.alpha)
    x_c   = _x_c(tpl)
    x_min = float(fit.xmin)
    x_max = float(arr.max())

    # CI for alpha: run-clustered bootstrap if run_ids are available and
    # requested, else the fast analytic Clauset formula.
    sigma = (alpha - 1.0) / np.sqrt(n_tail)
    ci_method = "analytic"
    ci_lo, ci_hi = alpha - 1.96 * sigma, alpha + 1.96 * sigma
    if run_ids is not None and n_bootstrap > 0:
        tail_run_ids = [rid for x, rid in zip(arr, ids) if x >= fit.xmin]
        boot_lo, boot_hi = bootstrap_alpha_ci(
            tail_data, tail_run_ids, x_min, is_discrete,
            n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        )
        if np.isfinite(boot_lo) and np.isfinite(boot_hi):
            ci_lo, ci_hi, ci_method = boot_lo, boot_hi, "run_clustered_bootstrap"

    # GOF for the fitted TPL: semiparametric bootstrap if requested, else
    # the fast KS<0.1 heuristic.
    ks_stat = _ks_distance(tpl)
    gof_method = "ks_heuristic"
    gof_p = float("nan") if np.isnan(ks_stat) else (1.0 if ks_stat < 0.1 else 0.0)
    if n_gof_sim > 0:
        sim_p = semiparametric_gof(fit, n_sim=n_gof_sim, seed=bootstrap_seed)
        if np.isfinite(sim_p):
            gof_p, gof_method = sim_p, "semiparametric_bootstrap"

    # Non-nested Vuong tests: TPL vs LN, TPL vs exponential
    def _lr(dist: str):
        try:
            R, p = fit.distribution_compare("truncated_power_law", dist,
                                            normalized_ratio=True)
            return float(R), float(p)
        except Exception:
            return float("nan"), float("nan")

    lr_ln,  p_ln  = _lr("lognormal")
    lr_exp, p_exp = _lr("exponential")

    # Boundary-aware nested LRT: TPL vs PL
    lr_tpl_pl, p_tpl_pl = boundary_aware_lrt(fit)

    # Gini on tail data only — consistent with alpha/KS/LR which are tail-only
    gini = _gini(tail_data)

    result = FitResult(
        observable=name,
        label=label,
        n_total=len(arr),
        n_tail=n_tail,
        tail_fraction=round(n_tail / len(arr), 4),
        x_min=round(x_min, 2),
        x_max=round(x_max, 1),
        mean=round(float(arr.mean()), 3),
        median=round(float(np.median(arr)), 3),
        dynamic_range=dynamic_range,
        alpha=round(alpha, 4),
        x_c=round(x_c, 2) if np.isfinite(x_c) else float("inf"),
        sigma_alpha=round(sigma, 4),
        alpha_ci_low=round(ci_lo, 4) if np.isfinite(ci_lo) else float("nan"),
        alpha_ci_high=round(ci_hi, 4) if np.isfinite(ci_hi) else float("nan"),
        ci_method=ci_method,
        ks_stat=round(ks_stat, 4) if not np.isnan(ks_stat) else float("nan"),
        gof_pvalue=round(gof_p, 4) if not np.isnan(gof_p) else float("nan"),
        gof_method=gof_method,
        lr_tpl_vs_ln=round(lr_ln, 4),
        lr_tpl_vs_ln_p=round(p_ln, 4),
        lr_tpl_vs_exp=round(lr_exp, 4),
        lr_tpl_vs_exp_p=round(p_exp, 4),
        lr_tpl_vs_pl=round(lr_tpl_pl, 4) if np.isfinite(lr_tpl_pl) else float("nan"),
        lr_tpl_vs_pl_p=round(p_tpl_pl, 4) if np.isfinite(p_tpl_pl) else float("nan"),
        tpl_plausible=(gof_p > 0.1 if not np.isnan(gof_p) else False),
        tpl_beats_lognormal=(lr_ln > 0 and p_ln < 0.05),
        tpl_beats_exponential=(lr_exp > 0 and p_exp < 0.05),
        tpl_beats_pl=(lr_tpl_pl > 0 and p_tpl_pl < 0.05) if np.isfinite(p_tpl_pl) else False,
        regime=_regime(alpha),
        gini=round(gini, 4),
    )

    if verbose:
        _print_result(result)

    return result


def _print_result(r: FitResult) -> None:
    tpl_ok = "✓" if r.tpl_plausible else "✗"
    ln_ok  = "✓" if r.tpl_beats_lognormal else "✗"
    xc_str = f"{r.x_c:.1f}" if np.isfinite(r.x_c) else "inf"
    print(
        f"  {r.label:<30}  n_tail={r.n_tail:5d} ({r.tail_fraction:.1%})  "
        f"xmin={r.x_min:6.1f}  α={r.alpha:.3f}[{r.alpha_ci_low:.2f},{r.alpha_ci_high:.2f}]  "
        f"x_c={xc_str}  "
        f"GOF={r.gof_pvalue:.3f}{tpl_ok}  "
        f"R_TPL/LN={r.lr_tpl_vs_ln:+.2f}(p={r.lr_tpl_vs_ln_p:.3f}){ln_ok}  "
        f"2Δℓ_TPL/PL={r.lr_tpl_vs_pl:.2f}(p={r.lr_tpl_vs_pl_p:.3f})  "
        f"regime={r.regime}  dyn={r.dynamic_range}"
    )


# ── Batch fitting ─────────────────────────────────────────────

def fit_all(
    event_observables: Dict[str, List[int]],
    run_ids:            Optional[Dict[str, List[str]]] = None,
    verbose:             bool = True,
    n_bootstrap:          int = 50,
    n_gof_sim:            int = 0,
    specs:                Optional[Dict[str, dict]] = None,
    min_n_total:          int = 50,
    min_n_tail:           int = 50,
) -> Dict[str, FitResult]:
    """
    Fit all observables from extract_all_observables()["event_observables"].
    Pool data across seeds before calling this — do not fit per-run.

    run_ids, if provided, must have the same keys as event_observables, each
    a list parallel to that observable's data giving the source run id — used
    for run-clustered bootstrap CIs.

    specs, min_n_total, min_n_tail are passed through to fit_observable() —
    see there (specs overrides OBSERVABLES; min_n_* relax the sample-size
    floor, e.g. for smoke tests).

    Holm-corrects lr_tpl_vs_ln_p and lr_tpl_vs_pl_p across the returned
    observables to control the family-wise error rate.
    """
    if verbose:
        print("Truncated power-law fitting (discrete MLE, Clauset-Newman-Shalizi):")
    results: Dict[str, FitResult] = {}
    for name, data in event_observables.items():
        rids = (run_ids or {}).get(name)
        result = fit_observable(
            data, name=name, verbose=verbose, run_ids=rids,
            n_bootstrap=n_bootstrap, n_gof_sim=n_gof_sim,
            specs=specs, min_n_total=min_n_total, min_n_tail=min_n_tail,
        )
        if result is not None:
            results[name] = result

    if len(results) > 1:
        names = list(results.keys())
        ln_p_adj  = holm_correction([results[n].lr_tpl_vs_ln_p for n in names])
        pl_p_adj  = holm_correction([results[n].lr_tpl_vs_pl_p for n in names])
        for n, p_ln, p_pl in zip(names, ln_p_adj, pl_p_adj):
            results[n].tags.append(f"holm_lr_tpl_vs_ln_p={p_ln:.4f}")
            results[n].tags.append(f"holm_lr_tpl_vs_pl_p={p_pl:.4f}")

    return results


# ── DataFrame export ──────────────────────────────────────────

def fits_to_dataframe(fits: Dict[str, FitResult]):
    """Convert to a pandas DataFrame for LaTeX/CSV export."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for fits_to_dataframe()")

    rows = []
    for name, r in fits.items():
        rows.append({
            "Observable":    r.label,
            "n_tail":        r.n_tail,
            "tail_frac":     r.tail_fraction,
            "x_min":         r.x_min,
            "x_max":         r.x_max,
            "dyn_range":     r.dynamic_range,
            "alpha":         r.alpha,
            "alpha_ci":      f"[{r.alpha_ci_low:.2f}, {r.alpha_ci_high:.2f}]",
            "x_c":           r.x_c,
            "GOF_p":         r.gof_pvalue,
            "TPL_ok":        "✓" if r.tpl_plausible else "✗",
            "R_TPL_LN":      r.lr_tpl_vs_ln,
            "R_TPL_LN_p":    r.lr_tpl_vs_ln_p,
            "TPL>LN":        "✓" if r.tpl_beats_lognormal else "✗",
            "2dl_TPL_PL":    r.lr_tpl_vs_pl,
            "2dl_TPL_PL_p":  r.lr_tpl_vs_pl_p,
            "TPL>PL":        "✓" if r.tpl_beats_pl else "✗",
            "Regime":        r.regime,
            "Gini_tail":     r.gini,
        })
    return pd.DataFrame(rows)


# ── CCDF helpers (used by visualization/ccdf_panel.py) ───────

def empirical_ccdf(data: List[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, P(X >= x)) for plotting.
    P(X >= x) = (n - rank) / n, where rank is 0-indexed ascending.
    """
    arr = np.sort(np.array([x for x in data if x > 0], dtype=float))
    n   = len(arr)
    p   = (n - np.arange(n)) / n
    return arr, p


def powerlaw_ccdf_line(
    x_min:    float,
    x_max:    float,
    alpha:    float,
    x_c:      float = float("inf"),
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Theoretical CCDF for the fitted model, normalised to 1 at x_min.
    Truncated power law by default (x_c=inf reduces to pure PL).

    For the TPL case, alpha is typically > 1, making the survival integral's
    incomplete-gamma order negative — scipy.special.gammaincc only supports
    positive order, so the tail integral P(X>=x) ~ int_x^inf t^-alpha e^{-t/x_c} dt
    is evaluated directly via numerical quadrature instead (robust for any alpha).
    """
    x = np.logspace(np.log10(max(x_min, 1)), np.log10(x_max), n_points)
    if np.isfinite(x_c) and x_c > 0:
        from scipy.integrate import quad
        integrand = lambda t: t ** (-alpha) * np.exp(-t / x_c)
        norm = quad(integrand, x_min, np.inf)[0]
        ccdf = np.array([quad(integrand, xi, np.inf)[0] / norm for xi in x])
    else:
        ccdf = (x / x_min) ** (-(alpha - 1.0))
    return x, ccdf
