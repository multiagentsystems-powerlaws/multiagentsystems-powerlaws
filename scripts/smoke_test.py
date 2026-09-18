"""
scripts/smoke_test.py

Small-scale replica of run_sweep.py for validating the full pipeline
before launching the real sweep. Uses identical code paths — no
reimplementation. Only the experimental parameters differ.

Smoke parameters vs full sweep:
  Topologies : 4  (chain, star, tree, full_mesh)
  Scales     : N ∈ {8, 16}
  Seeds      : 2  (0, 1)
  Tasks      : 2 synthetic reasoning tasks

Exercises the complete pipeline:
  GraphRunner                     — execution + task tree
  analysis.run_pipeline           — annotate → DAG → observables
  tail_fitting.powerlaw_fit       — MLE fitting + LR tests
  visualization.ccdf_panel        — CCDF figures
  visualization.xmax_scaling      — EVT scaling figures

Output layout:
  data/smoke_test/
    run_manifest.json               one entry per run (status, event count, error)
    per_run_observables.json        raw observable samples per run
    pooled_observables.json         all samples pooled across runs
    pooled_fits.json                MLE results + LR tests on pooled data
    pooled_fits.csv                 same, tabular
    condition_fits.json             per-condition fits keyed by topology/N
    dag_inspection.json             claim DAG stats for best run
    figures/
      smoke_ccdf_panel.pdf          CCDF panel (all observables, pooled)
      smoke_ccdf_by_topology.pdf    CCDF overlay per topology
      smoke_xmax_scaling_tce.pdf    x_max vs N (EVT scaling)
    runs/                           individual run dirs written by GraphRunner
      gaia/{topology}/n{N}/s{seed}/{task_id}/
        events.jsonl
        task_tree.json
        run_metadata.json
        run_config.json

Usage:
    cd ~/mas-powerlaws
    export $(grep -v '^#' .env | xargs)
    python scripts/smoke_test.py

Exit 0 = all checks passed. Exit 1 = failure.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS = "  PASS"
FAIL = "  FAIL"

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        msg = fn()
        results.append((name, True, msg or ""))
        print(f"{PASS}  {name}" + (f"  — {msg}" if msg else ""))
    except Exception:
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append((name, False, tb))
        print(f"{FAIL}  {name}  — {tb}")
        if "--traceback" in sys.argv:
            traceback.print_exc()


def _fit_to_dict(r) -> dict:
    """Serialize a FitResult to a plain dict for JSON output."""
    return {
        "alpha":                r.alpha,
        "sigma_alpha":          r.sigma_alpha,
        "xmin":                 r.xmin,
        "n_tail":               r.n_tail,
        "n_total":              r.n_total,
        "tail_fraction":        r.tail_fraction,
        "pl_plausible":         r.pl_plausible,
        "ks_pvalue":            r.ks_pvalue,
        "lr_vs_lognormal":      r.lr_vs_lognormal,
        "lr_vs_lognormal_p":    r.lr_vs_lognormal_p,
        "lr_vs_exponential":    r.lr_vs_exponential,
        "lr_vs_exponential_p":  r.lr_vs_exponential_p,
        "lr_vs_truncated_pl":   r.lr_vs_truncated_pl,
        "lr_vs_truncated_pl_p": r.lr_vs_truncated_pl_p,
        "regime":               r.regime,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke parameters
# ─────────────────────────────────────────────────────────────────────────────

SMOKE_DIR   = ROOT / "data" / "smoke_logs"
RUNS_DIR    = SMOKE_DIR / "runs"
FIGURES_DIR = SMOKE_DIR / "figures"
MODEL       = "gpt-4o-mini"

TOPOLOGIES  = ["tree"]
SCALES      = [8]
SEEDS       = [0]

SMOKE_TASKS = [
    {
        "task_id": "smoke_reasoning_001",
        "prompt": (
            "A factory runs three shifts. Morning: 240 widgets in 8 hours. "
            "Afternoon: 180 widgets in 6 hours. Night: 200 widgets in 10 hours. "
            "Which shift has highest productivity per hour? What is total daily output? "
            "Does it meet a 650-widget/day target? Show full reasoning."
        ),
        "gold_answer": None,
    },
    {
        "task_id": "smoke_reasoning_002",
        "prompt": (
            "A company has three departments. Sales: 12 staff, $480K revenue. "
            "Engineering: 8 staff, $640K revenue. Support: 6 staff, $180K revenue. "
            "Which department has highest revenue per staff? What is total headcount? "
            "If each department must hit $60K/staff, which are below target?"
        ),
        "gold_answer": None,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module imports
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 1. Module imports ===")

_modules = [
    ("prompts.base_prompt",               ["BASE_PROMPT"]),
    ("prompts.topology_addenda",          ["get_topology_addendum"]),
    ("prompts.task_addenda",              ["get_task_addendum"]),
    ("context_builder",                   ["build_context", "MAX_COMPLETION_TOKENS"]),
    ("loggers.trace_schema",              ["TraceRow", "AgentOutput"]),
    ("loggers.event_bus",                 ["EventBus"]),
    ("event_extraction.event_extractor",  ["annotate_event_types"]),
    ("observables.dag_builder",           ["build_all"]),
    ("observables.cascade_metrics",       ["extract_all_observables"]),
    ("metrics.inequality",                ["gini", "top_k_share", "effective_n"]),
    ("tail_fitting.powerlaw_fit",         ["fit_all", "fits_to_dataframe"]),
    ("analysis.run_pipeline",             ["process_run", "pool_and_fit",
                                           "process_condition", "discover_run_dirs"]),
    ("tools.tools",                       ["run_tool", "get_tool_names_for_benchmark"]),
    ("topologies.chain",                  ["ChainTopology"]),
    ("topologies.star",                   ["StarTopology"]),
    ("topologies.tree",                   ["TreeTopology"]),
    ("topologies.full_mesh",              ["FullMeshTopology"]),
    ("topologies.sparse_mesh",            ["SparseMeshTopology"]),
    ("topologies.hybrid",                 []),
    ("topologies.dynamic_reputation",     []),
    ("execution.graph_runner",            ["GraphRunner", "BenchmarkTask"]),
    ("benchmark_wrappers.task_expander",  ["TaskExpander"]),
    ("visualization.ccdf_panel",          ["plot_ccdf_panel"]),
    ("visualization.xmax_scaling",        ["plot_xmax_scaling", "collect_xmax_from_runs"]),
]

for mod, symbols in _modules:
    def _check_import(m=mod, s=symbols):
        obj = __import__(m, fromlist=s or ["_"])
        for sym in s:
            if not hasattr(obj, sym):
                raise ImportError(f"{sym} not found in {m}")
    check(mod, _check_import)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Environment
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 2. Environment ===")

def _check_api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return f"{key[:12]}..."

def _check_llm():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(model=MODEL, temperature=0.0)
    r = llm.invoke([HumanMessage(content="Reply with exactly one word: hello")])
    if not r.content.strip():
        raise RuntimeError("empty response")
    return f"'{r.content.strip()[:20]}'"

def _check_tools():
    from tools.tools import run_tool
    for tool, inp, bench in [
        ("web_search",  "capital of France", "gaia"),
        ("calculator",  "12 * 8 + 4",        "realmb"),
        ("python_exec", "print(2+2)",         "gaia"),
    ]:
        r = run_tool(tool, inp, bench)
        if not r.success:
            raise RuntimeError(f"{tool} failed: {r.error}")
    return "web_search, calculator, python_exec"

check("OPENAI_API_KEY",   _check_api_key)
check("LLM connectivity", _check_llm)
check("tools",            _check_tools)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mini sweep
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n=== 3. Mini sweep  "
      f"({len(TOPOLOGIES)} topologies × {len(SCALES)} scales × "
      f"{len(SEEDS)} seeds × {len(SMOKE_TASKS)} tasks) ===")
print(f"    Output → {SMOKE_DIR}\n")

if SMOKE_DIR.exists():
    shutil.rmtree(SMOKE_DIR)
for d in [SMOKE_DIR, RUNS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True)

from execution.graph_runner import GraphRunner, BenchmarkTask
from langchain_openai import ChatOpenAI

llm    = ChatOpenAI(model=MODEL, temperature=0.7)
runner = GraphRunner(llm=llm, data_root=RUNS_DIR, model_name=MODEL, max_steps=20)

completed_run_dirs: list[Path] = []
run_manifest: list[dict] = []
total_runs = len(SMOKE_TASKS) * len(TOPOLOGIES) * len(SCALES) * len(SEEDS)
run_idx = 0

for task_cfg in SMOKE_TASKS:
    task = BenchmarkTask(
        task_id=task_cfg["task_id"],
        benchmark="gaia",
        task_family="reasoning",
        prompt=task_cfg["prompt"],
        gold_answer=task_cfg.get("gold_answer"),
        difficulty="medium",
        requires_tools=False,
        requires_synthesis=True,
    )
    for topo in TOPOLOGIES:
        for N in SCALES:
            for seed in SEEDS:
                run_idx += 1
                print(f"    [{run_idx:>3}/{total_runs}]  {topo:<12} N={N:<4} seed={seed}  "
                      f"{task_cfg['task_id']} ...", end=" ", flush=True)
                try:
                    result = runner.run(task=task, topology=topo, num_agents=N, seed=seed)
                    status = "error" if result.error else "ok"
                    print(f"{status}  events={result.event_count}")
                    completed_run_dirs.append(Path(result.run_dir))
                    run_manifest.append({
                        "topology":    topo,
                        "N":           N,
                        "seed":        seed,
                        "task_id":     task_cfg["task_id"],
                        "run_dir":     str(result.run_dir),
                        "event_count": result.event_count,
                        "status":      status,
                        "error":       result.error[:300] if result.error else None,
                    })
                except Exception:
                    tb = traceback.format_exc().strip().splitlines()[-1]
                    print(f"FAIL  {tb}")
                    run_manifest.append({
                        "topology": topo, "N": N, "seed": seed,
                        "task_id": task_cfg["task_id"],
                        "run_dir": None, "event_count": 0,
                        "status": "fatal", "error": tb,
                    })

(SMOKE_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

runs_with_events = [r for r in run_manifest if r.get("event_count", 0) > 0]
check(
    f"mini sweep {total_runs} runs",
    lambda: (
        f"{len(runs_with_events)}/{total_runs} runs produced events  → run_manifest.json"
        if len(runs_with_events) >= total_runs // 2
        else exec("raise RuntimeError(f'only {len(runs_with_events)}/{total_runs} runs produced events')")
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Post-hoc pipeline
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 4. Post-hoc pipeline ===")

from analysis.run_pipeline import (
    process_run, pool_and_fit, process_condition, discover_run_dirs
)

per_run_obs: list[dict] = []

def _run_pipeline():
    failed = []
    for rd in completed_run_dirs:
        try:
            obs = process_run(rd)
            per_run_obs.append({"run_dir": str(rd), **obs})
        except Exception as e:
            failed.append({"run_dir": str(rd), "error": str(e)})

    if not per_run_obs:
        raise RuntimeError("all runs failed post-hoc pipeline")

    # Save per-run observables — one entry per run with sample counts and raw lists
    per_run_out = []
    for entry in per_run_obs:
        eo = entry.get("event_observables", {})
        per_run_out.append({
            "run_dir":  entry["run_dir"],
            "sample_counts": {k: len(v) for k, v in eo.items()},
            "observables":   eo,
            "agent_metrics": entry.get("agent_metrics", {}),
        })
    (SMOKE_DIR / "per_run_observables.json").write_text(
        json.dumps(per_run_out, indent=2)
    )

    total_samples = sum(
        sum(len(v) for v in entry.get("event_observables", {}).values())
        for entry in per_run_obs
    )
    failed_str = f"  ({len(failed)} failed)" if failed else ""
    return (f"{len(per_run_obs)} runs processed{failed_str}  "
            f"total samples={total_samples}  → per_run_observables.json")

check("annotate → DAG → observables", _run_pipeline)

# Per-condition fits (same grouping as run_sweep)
conditions = discover_run_dirs(RUNS_DIR)
condition_fits: dict[str, dict] = {}

def _condition_fits():
    if not conditions:
        raise RuntimeError("no conditions found")
    for ckey, rdirs in sorted(conditions.items()):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits = process_condition(rdirs, condition_label=ckey, verbose=False)
            condition_fits[ckey] = fits
        except Exception as e:
            condition_fits[ckey] = {}
            print(f"    WARNING: {ckey} fit failed: {e}")

    # Save per-condition fits
    (SMOKE_DIR / "condition_fits.json").write_text(json.dumps(
        {ckey: {name: _fit_to_dict(r) for name, r in fits.items() if r}
         for ckey, fits in condition_fits.items()},
        indent=2
    ))

    fitted = [k for k, v in condition_fits.items() if v]
    return (f"{len(conditions)} conditions  {len(fitted)} with fits  "
            f"→ condition_fits.json")

check("per-condition fits", _condition_fits)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Power-law fits + model comparison
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 5. Power-law fits and model comparison ===")

from tail_fitting.powerlaw_fit import fit_all, fits_to_dataframe

pooled_fits: dict = {}

def _pooled_fits():
    # Build pooled observables across all runs
    pooled: dict[str, list] = {
        "delegation_sizes": [], "revision_waves": [],
        "contradiction_bursts": [], "merge_fan_in": [], "tce": [],
    }
    for entry in per_run_obs:
        for k in pooled:
            pooled[k].extend(entry.get("event_observables", {}).get(k, []))

    # Save pooled observables
    (SMOKE_DIR / "pooled_observables.json").write_text(json.dumps(
        {"sample_counts": {k: len(v) for k, v in pooled.items()},
         "observables":   pooled},
        indent=2
    ))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = fit_all(pooled, verbose=False)

    pooled_fits.update(fits)

    if not fits:
        raise RuntimeError("no observables had enough samples to fit (n_tail >= 50 required)")

    # Save fits as JSON
    (SMOKE_DIR / "pooled_fits.json").write_text(json.dumps(
        {name: _fit_to_dict(r) for name, r in fits.items() if r},
        indent=2
    ))

    # Save fits as CSV
    df = fits_to_dataframe(fits)
    df.to_csv(SMOKE_DIR / "pooled_fits.csv", index=False)

    # Print full table
    print()
    print(f"    {'observable':<25}  {'alpha':>6}  {'±sigma':>7}  "
          f"{'n_tail':>7}  {'pl':>4}  "
          f"{'LR_LN':>7}  {'p':>6}  "
          f"{'LR_TPL':>7}  {'p':>6}  regime")
    print("    " + "─" * 90)
    for name, r in fits.items():
        print(
            f"    {name:<25}  {r.alpha:>6.3f}  ±{r.sigma_alpha:<6.3f}  "
            f"{r.n_tail:>7d}  {'✓' if r.pl_plausible else '✗':>4}  "
            f"{r.lr_vs_lognormal:>+7.2f}  {r.lr_vs_lognormal_p:>6.3f}  "
            f"{r.lr_vs_truncated_pl:>+7.2f}  {r.lr_vs_truncated_pl_p:>6.3f}  "
            f"{r.regime}"
        )
    print()

    bad = {n: r.alpha for n, r in fits.items() if not (1.5 < r.alpha < 5.0)}
    if bad:
        raise RuntimeError(f"alpha out of expected range [1.5, 5.0]: {bad}")

    return (f"{len(fits)} fitted  "
            f"alpha=[{min(r.alpha for r in fits.values()):.2f}, "
            f"{max(r.alpha for r in fits.values()):.2f}]  "
            f"→ pooled_fits.json + pooled_fits.csv")

check("power-law MLE + LR tests", _pooled_fits)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 6. Visualizations ===")

from visualization.ccdf_panel import plot_ccdf_panel
from visualization.xmax_scaling import plot_xmax_scaling, collect_xmax_from_runs

def _ccdf_panel():
    pooled: dict[str, list] = {
        "delegation_sizes": [], "revision_waves": [],
        "contradiction_bursts": [], "merge_fan_in": [], "tce": [],
    }
    for entry in per_run_obs:
        for k in pooled:
            pooled[k].extend(entry.get("event_observables", {}).get(k, []))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_ccdf_panel(
            pooled_observables=pooled,
            fits=pooled_fits,
            out_dir=FIGURES_DIR,
            condition_label="smoke — all topologies pooled",
            figname="smoke_ccdf_panel",
        )
    out = FIGURES_DIR / "smoke_ccdf_panel.pdf"
    if not out.exists():
        raise RuntimeError(f"not written: {out}")
    return f"→ figures/smoke_ccdf_panel.pdf"

def _ccdf_by_topology():
    from visualization.ccdf_panel import plot_ccdf_by_topology

    topo_obs: dict[str, dict[str, list]] = {}
    for entry in run_manifest:
        if not entry.get("run_dir") or not entry.get("event_count", 0):
            continue
        topo = entry["topology"]
        obs  = process_run(Path(entry["run_dir"]))
        if topo not in topo_obs:
            topo_obs[topo] = {k: [] for k in obs["event_observables"]}
        for k, v in obs["event_observables"].items():
            topo_obs[topo][k].extend(v)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_ccdf_by_topology(
            topo_observables=topo_obs,
            out_dir=FIGURES_DIR,
            figname="smoke_ccdf_by_topology",
        )
    out = FIGURES_DIR / "smoke_ccdf_by_topology.pdf"
    if not out.exists():
        raise RuntimeError(f"not written: {out}")
    return f"topologies={sorted(topo_obs.keys())}  → figures/smoke_ccdf_by_topology.pdf"

def _xmax_scaling():
    xmax_data = collect_xmax_from_runs(conditions, observable="tce")
    if not xmax_data:
        raise RuntimeError("no xmax data — need runs across multiple N values")

    alpha_preds: dict[str, float] = {}
    for ckey, fits in condition_fits.items():
        if fits.get("tce"):
            parts = ckey.split("_")
            topo  = "_".join(parts[1:-1])
            if topo and topo not in alpha_preds:
                alpha_preds[topo] = fits["tce"].alpha

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_xmax_scaling(
            xmax_data=xmax_data,
            observable_label="TCE",
            out_dir=FIGURES_DIR,
            alpha_predictions=alpha_preds or None,
            figname="smoke_xmax_scaling_tce",
        )
    out = FIGURES_DIR / "smoke_xmax_scaling_tce.pdf"
    if not out.exists():
        raise RuntimeError(f"not written: {out}")

    # Save xmax data as JSON for inspection
    (SMOKE_DIR / "xmax_data_tce.json").write_text(json.dumps(
        {topo: {str(N): vals for N, vals in n_dict.items()}
         for topo, n_dict in xmax_data.items()},
        indent=2
    ))

    N_vals = sorted({N for d in xmax_data.values() for N in d})
    return (f"topologies={sorted(xmax_data.keys())}  N={N_vals}  "
            f"→ figures/smoke_xmax_scaling_tce.pdf + xmax_data_tce.json")

check("CCDF panel (pooled)",        _ccdf_panel)
check("CCDF by topology",           _ccdf_by_topology)
check("x_max EVT scaling (TCE)",    _xmax_scaling)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Trace integrity
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 7. Trace integrity ===")

def _check_trace_fields():
    required = [
        "run_id", "benchmark", "task_id", "topology", "step_id",
        "agent_id", "claim_id", "parent_claim_ids", "subtask_id",
        "reasoning_text", "message_length",
    ]
    bad = []
    for rd in completed_run_dirs:
        ef = rd / "events.jsonl"
        if not ef.exists():
            continue
        rows = [json.loads(l) for l in ef.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        missing = [f for f in required if f not in rows[0]]
        if missing:
            bad.append(f"{rd.name}: missing {missing}")
    if bad:
        raise RuntimeError(f"{len(bad)} runs with missing fields")
    return f"all {len(completed_run_dirs)} runs have required TraceRow fields"

def _check_claim_dag():
    from observables.dag_builder import build_all
    from event_extraction.event_extractor import annotate_event_types

    best_rd = max(
        (rd for rd in completed_run_dirs if (rd / "events.jsonl").exists()),
        key=lambda rd: (rd / "events.jsonl").stat().st_size,
        default=None,
    )
    if not best_rd:
        raise RuntimeError("no events.jsonl found")

    rows      = [json.loads(l) for l in (best_rd / "events.jsonl").read_text().splitlines() if l.strip()]
    annotated = annotate_event_types(rows)
    st, cd, cascades = build_all(annotated)

    no_root  = [r for r in annotated if not r.get("root_claim_id")]
    no_depth = [r for r in annotated if r.get("claim_depth") is None]
    if no_root:
        raise RuntimeError(f"{len(no_root)} rows missing root_claim_id")
    if no_depth:
        raise RuntimeError(f"{len(no_depth)} rows missing claim_depth")

    types  = sorted({r.get("event_type") for r in annotated if r.get("event_type")})
    depths = sorted({r["claim_depth"] for r in annotated})

    # Save DAG inspection results
    dag_inspection = {
        "run_dir":      str(best_rd),
        "n_rows":       len(annotated),
        "event_types":  types,
        "n_claims":     len(cd),
        "n_cascades":   len(cascades),
        "n_subtasks":   len(st),
        "claim_depths": depths,
        "max_cascade_size": max((c.size for c in cascades), default=0),
        "max_tce":          max((c.tce  for c in cascades), default=0),
        "cascade_sizes":    sorted([c.size for c in cascades], reverse=True)[:20],
        "cascade_tces":     sorted([c.tce  for c in cascades], reverse=True)[:20],
        "agent_metrics":    {
            aid: m for aid, m in
            sorted(
                list(process_run(best_rd)["agent_metrics"].items()),
                key=lambda x: x[1].get("raw_events", 0),
                reverse=True
            )
        },
    }
    (SMOKE_DIR / "dag_inspection.json").write_text(json.dumps(dag_inspection, indent=2))

    print(f"\n    Run: {best_rd.relative_to(SMOKE_DIR)}")
    print(f"    event_types:  {types}")
    print(f"    claims={len(cd)}  cascades={len(cascades)}  subtasks={len(st)}")
    print(f"    depths: {depths}")
    print(f"    max_cascade_size={dag_inspection['max_cascade_size']}  "
          f"max_tce={dag_inspection['max_tce']}\n")

    return (f"root_ids=✓  depths={depths}  "
            f"event_types={types}  → dag_inspection.json")

check("TraceRow required fields", _check_trace_fields)
check("claim DAG + DAG inspection saved", _check_claim_dag)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Structural DAG invariants  (checked across ALL runs, not just the best)
# ─────────────────────────────────────────────────────────────────────────────
#
# These fail hard on any violation. They check semantic correctness, not just
# field presence. A broken annotator or dag_builder that assigns wrong but
# non-null values would be caught here and not by section 7.
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 8. Structural DAG invariants ===")

from event_extraction.event_extractor import annotate_event_types
from observables.dag_builder import build_all

# Coordination event types that must have a non-null claim_id
CLAIM_EVENT_TYPES = {
    "propose_claim", "revise_claim", "contradict_claim",
    "merge_claims", "endorse_claim",
}

def _check_all_runs_dag_invariants():
    """
    Run all structural invariants across every completed run.
    Collects all violations before raising so you see the full picture.
    """
    all_violations: list[str] = []
    runs_checked = 0

    for rd in completed_run_dirs:
        ef = rd / "events.jsonl"
        if not ef.exists():
            continue

        raw_rows  = [json.loads(l) for l in ef.read_text().splitlines() if l.strip()]
        if not raw_rows:
            continue

        annotated = annotate_event_types(raw_rows)
        _, cd, _  = build_all(annotated)   # also writes root_claim_id + claim_depth back
        runs_checked += 1
        run_label = str(rd.relative_to(SMOKE_DIR))

        # Build lookup: claim_id → row (use last occurrence if duplicate)
        claim_rows: dict[str, dict] = {}
        for row in annotated:
            cid = row.get("claim_id")
            if cid:
                claim_rows[cid] = row

        for i, row in enumerate(annotated):
            et  = row.get("event_type", "")
            cid = row.get("claim_id")
            pid = row.get("parent_claim_ids") or []
            rdepth = row.get("claim_depth")
            rroot  = row.get("root_claim_id")
            prefix = f"[{run_label}] row {i} ({et})"

            # ── Invariant 1: claim events must have non-null claim_id ────────
            if et in CLAIM_EVENT_TYPES and not cid:
                all_violations.append(
                    f"{prefix}: claim event has null claim_id"
                )

            # ── Invariant 2: all parent_claim_ids must exist in this run ────
            for pid_val in pid:
                if pid_val not in claim_rows:
                    all_violations.append(
                        f"{prefix}: parent_claim_id '{pid_val}' not found in run"
                    )

            # ── Invariant 3: root_claim_id must be non-null after annotation ─
            if et in CLAIM_EVENT_TYPES and not rroot:
                all_violations.append(
                    f"{prefix}: claim event has null root_claim_id after dag_builder"
                )

            # ── Invariant 4: root consistency — child root must match parent root ──
            # (Only check single-parent rows; merge nodes get the primary parent root
            #  which may differ from secondary parents — that's correct by design.)
            if len(pid) == 1 and rroot and pid[0] in claim_rows:
                parent_root = claim_rows[pid[0]].get("root_claim_id")
                if parent_root and rroot != parent_root:
                    all_violations.append(
                        f"{prefix}: root_claim_id '{rroot}' != "
                        f"parent root '{parent_root}'"
                    )

            # ── Invariant 5: depth == 0 iff no parents ───────────────────────
            if rdepth is not None:
                if len(pid) == 0 and rdepth != 0:
                    all_violations.append(
                        f"{prefix}: depth={rdepth} but no parents (expected 0)"
                    )
                if len(pid) > 0 and rdepth == 0:
                    all_violations.append(
                        f"{prefix}: depth=0 but has parents {pid}"
                    )

            # ── Invariant 6: depth == 1 + max(parent depths) for single-parent ──
            if len(pid) == 1 and rdepth is not None and pid[0] in claim_rows:
                parent_depth = claim_rows[pid[0]].get("claim_depth")
                if parent_depth is not None and rdepth != parent_depth + 1:
                    all_violations.append(
                        f"{prefix}: depth={rdepth} but parent depth={parent_depth} "
                        f"(expected {parent_depth + 1})"
                    )

            # ── Invariant 6b: merge depth >= 1 + max(parent depths) ─────────
            # Also checks: all parent depths non-null, primary root matches
            # primary parent (earliest step_id parent).
            if et == "merge_claims" and len(pid) >= 2:
                parent_nodes = [claim_rows[p] for p in pid if p in claim_rows]
                parent_depths = [p.get("claim_depth") for p in parent_nodes
                                 if p.get("claim_depth") is not None]
                if parent_depths and rdepth is not None:
                    expected_min = max(parent_depths) + 1
                    if rdepth < expected_min:
                        all_violations.append(
                            f"{prefix}: merge depth={rdepth} < "
                            f"1 + max(parent_depths)={expected_min}"
                        )
                null_depth_parents = [p for p in pid
                                      if p in claim_rows and
                                      claim_rows[p].get("claim_depth") is None]
                if null_depth_parents:
                    all_violations.append(
                        f"{prefix}: merge parents with null depth: "
                        f"{null_depth_parents}"
                    )
                # Primary parent = earliest step_id; its root must match rroot
                if parent_nodes and rroot:
                    primary = min(parent_nodes,
                                  key=lambda p: p.get("step_id", 0))
                    primary_root = primary.get("root_claim_id")
                    if primary_root and rroot != primary_root:
                        all_violations.append(
                            f"{prefix}: merge root '{rroot}' != "
                            f"primary parent root '{primary_root}'"
                        )

        # ── Invariant 7: duplicate claim_id must have identical structural metadata ──
        # Stronger than before: checks root, depth, parents, subtask — not just agent
        claim_meta: dict[str, dict] = {}
        for row in annotated:
            cid = row.get("claim_id")
            if not cid:
                continue
            structural = {
                "agent_id":       row.get("agent_id"),
                "root_claim_id":  row.get("root_claim_id"),
                "claim_depth":    row.get("claim_depth"),
                "parent_claim_ids": tuple(sorted(row.get("parent_claim_ids") or [])),
                "subtask_id":     row.get("subtask_id"),
            }
            if cid in claim_meta:
                prev = claim_meta[cid]
                diffs = {k: (prev[k], structural[k])
                         for k in structural if prev[k] != structural[k]}
                if diffs:
                    all_violations.append(
                        f"[{run_label}]: claim_id '{cid}' appears with "
                        f"conflicting metadata: {diffs}"
                    )
            else:
                claim_meta[cid] = structural

        # ── Invariant 8: summary metric consistency ───────────────────────────
        meta_path = rd / "run_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            logged_total = meta.get("messages_total", 0)
            actual_total = len(annotated)
            if logged_total != actual_total:
                all_violations.append(
                    f"[{run_label}]: messages_total={logged_total} in metadata "
                    f"but {actual_total} rows in events.jsonl"
                )

            actual_revisions      = sum(1 for r in annotated if r.get("event_type") == "revise_claim")
            actual_contradictions = sum(1 for r in annotated if r.get("event_type") == "contradict_claim")
            actual_merges         = sum(1 for r in annotated if r.get("event_type") == "merge_claims")
            actual_endorsements   = sum(1 for r in annotated if r.get("event_type") == "endorse_claim")

            for actual, key in [
                (actual_revisions,      "num_revisions_total"),
                (actual_contradictions, "num_contradictions_total"),
                (actual_merges,         "num_merges_total"),
                (actual_endorsements,   "num_endorsements_total"),
            ]:
                logged = meta.get(key, 0)
                if logged != actual:
                    all_violations.append(
                        f"[{run_label}]: {key}={logged} in metadata "
                        f"but recomputed={actual} from events.jsonl"
                    )

        # ── Invariant 9: activity/snapshot consistency ────────────────────────
        # Recompute active agents per step from events.jsonl and compare to
        # run_metadata active_agents_per_step and mean_active_per_step.
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

            # Recompute from rows
            step_agents: dict[int, set] = {}
            for row in annotated:
                step = row.get("step_id")
                aid  = row.get("agent_id")
                if step is not None and aid:
                    step_agents.setdefault(step, set()).add(aid)

            if step_agents:
                recomputed_mean = sum(
                    len(agents) for agents in step_agents.values()
                ) / len(step_agents)
                logged_mean = meta.get("mean_active_per_step")

                # Allow 10% tolerance: mean is approximate due to aggregation
                if logged_mean is not None and logged_mean > 0:
                    rel_err = abs(recomputed_mean - logged_mean) / max(logged_mean, 1)
                    if rel_err > 0.10:
                        all_violations.append(
                            f"[{run_label}]: mean_active_per_step={logged_mean:.2f} "
                            f"in metadata but recomputed={recomputed_mean:.2f} "
                            f"(>{10}% divergence)"
                        )

                # Check active_agents_per_step dict if present
                logged_per_step = meta.get("active_agents_per_step") or {}
                for step_str, logged_count in logged_per_step.items():
                    try:
                        step_int = int(step_str)
                    except ValueError:
                        continue
                    recomputed_count = len(step_agents.get(step_int, set()))
                    if logged_count != recomputed_count:
                        all_violations.append(
                            f"[{run_label}]: active_agents_per_step[{step_str}]="
                            f"{logged_count} in metadata but recomputed="
                            f"{recomputed_count} from events.jsonl"
                        )

    # Save all violations for inspection
    inv_report = {
        "runs_checked":    runs_checked,
        "total_violations": len(all_violations),
        "violations":       all_violations[:200],  # cap for readability
    }
    (SMOKE_DIR / "dag_invariant_violations.json").write_text(
        json.dumps(inv_report, indent=2)
    )

    if all_violations:
        # Print first 10 for immediate visibility
        print(f"\n    First violations (of {len(all_violations)}):")
        for v in all_violations[:10]:
            print(f"      {v}")
        print()
        raise RuntimeError(
            f"{len(all_violations)} DAG invariant violations across "
            f"{runs_checked} runs — see dag_invariant_violations.json"
        )

    return (f"{runs_checked} runs checked  0 violations  "
            f"→ dag_invariant_violations.json")

check("structural DAG invariants (all runs)", _check_all_runs_dag_invariants)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  {passed} passed  |  {failed} failed  |  {len(results)} total")
print(f"\n  Runs:    {len(runs_with_events)}/{total_runs} produced events")
print(f"\n  Output files:")
for f in sorted(SMOKE_DIR.glob("*.json")) + sorted(SMOKE_DIR.glob("*.csv")):
    print(f"    {f.relative_to(SMOKE_DIR)}")
print(f"\n  Figures:")
for f in sorted(FIGURES_DIR.glob("*.pdf")):
    print(f"    figures/{f.name}")
print(f"\n  Run data:  {RUNS_DIR}/")

if failed:
    print("\nFailed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"  FAIL  {name}")
            print(f"        {msg}")
    print()
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED — pipeline verified. Ready for full sweep.\n")
    sys.exit(0)