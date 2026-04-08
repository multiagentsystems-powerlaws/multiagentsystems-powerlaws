"""
scripts/run_sweep.py

Full experimental sweep for mas-powerlaws.

Design (paper Section H):
  Topologies : 7  (chain, star, tree, full_mesh, sparse_mesh,
                   hybrid_modular, dynamic_reputation)
  Scales     : N ∈ {8, 16, 32, 64, 128}
  Seeds      : 3  (0, 1, 2)
  Benchmarks : GAIA (qa/reasoning), MARBLE (planning/coordination),
               REALM (planning), SWE-bench (coding)

Total runs  : 7 × 5 × 3 × tasks_per_benchmark
              (~98K events target across all runs)

Output layout:
  data/sweep/
    {benchmark}/{topology}/n{N}/s{seed}/{task_id}/
      events.jsonl         ← raw TraceRows (one per agent turn)
      task_tree.json       ← TaskExpander output
      run_metadata.json    ← H2 metrics + scores
      run_config.json      ← full run config
    sweep_manifest.jsonl   ← one line per completed run
    sweep_errors.jsonl     ← one line per failed run
    sweep_progress.json    ← live progress counters

Usage:
    cd ~/mas-powerlaws
    export $(grep -v '^#' .env | xargs)

    # Full sweep (all benchmarks, all topologies, all scales)
    python scripts/run_sweep.py

    # Dry run — print plan, no LLM calls
    python scripts/run_sweep.py --dry-run

    # Subset
    python scripts/run_sweep.py --benchmarks gaia --topologies chain star --scales 8 16 --seeds 0

    # Resume (skips run dirs that already have run_metadata.json)
    python scripts/run_sweep.py --resume

    # Parallel workers (process-level, safe for rate limits)
    python scripts/run_sweep.py --workers 4

Exit 0 = all runs completed (some may have errors — check sweep_errors.jsonl).
Exit 1 = fatal setup error before any run started.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ── Experiment grid ───────────────────────────────────────────────────────────

ALL_TOPOLOGIES = [
    "chain",
    "star",
    "tree",
    "full_mesh",
    "sparse_mesh",
    "hybrid_modular",
    "dynamic_reputation",
]

ALL_SCALES = [8, 16, 32, 64, 128]

ALL_SEEDS = [0, 1, 2]

# Tasks per benchmark: (benchmark_key, task_family, max_tasks)
# Update number of tasks as per requirements
BENCHMARK_CONFIGS = [
    ("gaia",       "qa",          20),
    ("gaia",       "reasoning",   20),
    ("marble",     "planning",    20),
    ("marble",     "coordination",10),
    ("realm",      "planning",    20),
    ("swebench",   "coding",      10),
]


# ── Run spec ─────────────────────────────────────────────────────────────────

@dataclass
class RunSpec:
    benchmark:   str
    task_family: str
    task_id:     str
    task_prompt: str
    gold_answer: Optional[str]
    topology:    str
    num_agents:  int
    seed:        int
    run_dir:     Path

    def key(self) -> str:
        return f"{self.benchmark}__{self.topology}__n{self.num_agents}__s{self.seed}__{self.task_id}"


# ── Task loading ──────────────────────────────────────────────────────────────

def _load_tasks(benchmark: str, task_family: str, max_tasks: int) -> List[Dict]:
    """
    Load tasks from the benchmark wrapper.
    Returns list of dicts with keys: task_id, prompt, gold_answer.
    Falls back to synthetic tasks if wrapper not available.
    """
    try:
        if benchmark == "gaia":
            from benchmark_wrappers.gaia import load_gaia_tasks
            tasks = load_gaia_tasks(max_tasks=max_tasks, task_family=task_family)
        elif benchmark == "marble":
            from benchmark_wrappers.multiagentbench import load_multiagentbench_tasks
            tasks = load_multiagentbench_tasks(max_tasks=max_tasks, task_family=task_family)
        elif benchmark == "realm":
            from benchmark_wrappers.realm_bench import load_realm_tasks
            tasks = load_realm_tasks(max_tasks=max_tasks, task_family=task_family)
        elif benchmark == "swebench":
            from benchmark_wrappers.swebench import load_swebench_tasks
            tasks = load_swebench_tasks(max_tasks=max_tasks)
        else:
            tasks = []

        # Normalize to common schema
        out = []
        for t in tasks:
            if hasattr(t, "task_id"):
                out.append({
                    "task_id":    t.task_id,
                    "prompt":     getattr(t, "prompt", "") or getattr(t, "description", ""),
                    "gold_answer": getattr(t, "gold_answer", None),
                })
            elif isinstance(t, dict):
                out.append({
                    "task_id":    t.get("task_id") or t.get("id", f"task_{len(out):04d}"),
                    "prompt":     t.get("prompt") or t.get("description", ""),
                    "gold_answer": t.get("gold_answer") or t.get("answer"),
                })
        return out[:max_tasks]

    except Exception as e:
        print(f"  WARNING: could not load {benchmark}/{task_family}: {e}")
        print(f"           falling back to {max_tasks} synthetic tasks")
        return [
            {
                "task_id":    f"synthetic_{benchmark}_{task_family}_{i:04d}",
                "prompt":     (
                    f"Synthetic {task_family} task #{i} for {benchmark}. "
                    "Analyze the problem, reason through it step by step, "
                    "and provide a complete answer."
                ),
                "gold_answer": None,
            }
            for i in range(max_tasks)
        ]


# ── Run one cell ──────────────────────────────────────────────────────────────

def _run_one(spec: RunSpec, model_name: str, dry_run: bool) -> Dict[str, Any]:
    """
    Execute one (task, topology, N, seed) cell.
    Called in a subprocess when workers > 1.
    Returns a summary dict written to sweep_manifest.jsonl.
    """
    if dry_run:
        return {
            "key":        spec.key(),
            "status":     "dry_run",
            "run_dir":    str(spec.run_dir),
            "timestamp":  _now(),
        }

    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")

        from langchain_openai import ChatOpenAI
        from execution.graph_runner import GraphRunner, BenchmarkTask

        llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        task = BenchmarkTask(
            task_id=spec.task_id,
            benchmark=spec.benchmark,
            task_family=spec.task_family,
            prompt=spec.task_prompt,
            gold_answer=spec.gold_answer,
            difficulty="medium",
            requires_tools=(spec.benchmark in ("gaia",)),
            requires_synthesis=(spec.task_family in ("synthesis", "planning", "coordination")),
        )

        runner = GraphRunner(
            llm=llm,
            data_root=spec.run_dir.parents[3],  # data/sweep
            model_name=model_name,
            max_steps=20,
        )

        result = runner.run(
            task=task,
            topology=spec.topology,
            num_agents=spec.num_agents,
            seed=spec.seed,
        )

        return {
            "key":          spec.key(),
            "status":       "ok" if not result.error else "error",
            "run_dir":      str(result.run_dir),
            "benchmark":    spec.benchmark,
            "task_family":  spec.task_family,
            "task_id":      spec.task_id,
            "topology":     spec.topology,
            "num_agents":   spec.num_agents,
            "seed":         spec.seed,
            "score":        result.score,
            "final_answer": (result.final_answer or "")[:200],
            "event_count":  result.event_count,
            "tokens_total": result.tokens_total,
            "wall_time_s":  result.wall_time_s,
            "error":        result.error[:500] if result.error else None,
            "timestamp":    _now(),
        }

    except Exception:
        tb = traceback.format_exc()
        return {
            "key":       spec.key(),
            "status":    "fatal",
            "run_dir":   str(spec.run_dir),
            "error":     tb[:1000],
            "timestamp": _now(),
        }


# ── Progress tracker ──────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int, sweep_dir: Path):
        self.total     = total
        self.done      = 0
        self.ok        = 0
        self.errors    = 0
        self.skipped   = 0
        self.t0        = time.time()
        self.path      = sweep_dir / "sweep_progress.json"

    def update(self, status: str):
        self.done += 1
        if status == "ok":        self.ok      += 1
        elif status == "skipped": self.skipped += 1
        else:                     self.errors  += 1
        self._write()

    def _write(self):
        elapsed = time.time() - self.t0
        rate    = self.done / elapsed if elapsed > 0 else 0
        eta_s   = (self.total - self.done) / rate if rate > 0 else 0
        d = {
            "total": self.total, "done": self.done,
            "ok": self.ok, "errors": self.errors, "skipped": self.skipped,
            "elapsed_s": round(elapsed), "eta_s": round(eta_s),
            "rate_per_min": round(rate * 60, 1),
            "updated": _now(),
        }
        self.path.write_text(json.dumps(d, indent=2))

    def print_line(self, spec: RunSpec, status: str, extra: str = ""):
        elapsed = time.time() - self.t0
        pct     = 100 * self.done / self.total if self.total else 0
        rate    = self.done / elapsed if elapsed > 0 else 0
        eta_m   = (self.total - self.done) / rate / 60 if rate > 0 else 0
        icon    = {"ok": "✓", "error": "✗", "skipped": "→", "fatal": "✗✗", "dry_run": "·"}.get(status, "?")
        print(
            f"  {icon} [{self.done:>5}/{self.total}  {pct:5.1f}%  ETA {eta_m:5.1f}m]  "
            f"{spec.topology:<20} N={spec.num_agents:<4} s={spec.seed}  "
            f"{spec.benchmark}/{spec.task_family}/{spec.task_id[:20]:<20}"
            + (f"  {extra}" if extra else ""),
            flush=True,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _already_done(run_dir: Path) -> bool:
    return (run_dir / "run_metadata.json").exists()


# ── Main ──────────────────────────────────────────────────────────────────────

def build_run_specs(
    benchmarks:  List[str],
    topologies:  List[str],
    scales:      List[int],
    seeds:       List[int],
    sweep_dir:   Path,
    resume:      bool,
) -> tuple[List[RunSpec], Dict[str, List[Dict]]]:
    """
    Build the full run grid and load tasks.
    Returns (specs, tasks_by_benchmark_family).
    """
    # Load tasks for each (benchmark, task_family) combo
    tasks_cache: Dict[str, List[Dict]] = {}
    bench_configs = [
        (b, tf, n) for b, tf, n in BENCHMARK_CONFIGS
        if b in benchmarks
    ]

    print("\nLoading benchmark tasks...")
    for benchmark, task_family, max_tasks in bench_configs:
        key = f"{benchmark}/{task_family}"
        if key not in tasks_cache:
            tasks = _load_tasks(benchmark, task_family, max_tasks)
            tasks_cache[key] = tasks
            print(f"  {key:<30} {len(tasks)} tasks")

    # Build run specs
    specs: List[RunSpec] = []
    for benchmark, task_family, _ in bench_configs:
        key   = f"{benchmark}/{task_family}"
        tasks = tasks_cache.get(key, [])
        for task in tasks:
            for topology in topologies:
                for N in scales:
                    for seed in seeds:
                        run_dir = (
                            sweep_dir / benchmark / topology
                            / f"n{N}" / f"s{seed}" / task["task_id"]
                        )
                        spec = RunSpec(
                            benchmark=benchmark,
                            task_family=task_family,
                            task_id=task["task_id"],
                            task_prompt=task["prompt"],
                            gold_answer=task.get("gold_answer"),
                            topology=topology,
                            num_agents=N,
                            seed=seed,
                            run_dir=run_dir,
                        )
                        if resume and _already_done(run_dir):
                            continue
                        specs.append(spec)

    return specs, tasks_cache


def main():
    parser = argparse.ArgumentParser(description="Run the full mas-powerlaws sweep.")
    parser.add_argument("--benchmarks",  nargs="+", default=[b for b, _, _ in BENCHMARK_CONFIGS],
                        choices=["gaia", "marble", "realm", "swebench"],
                        help="Benchmarks to run (default: all)")
    parser.add_argument("--topologies",  nargs="+", default=ALL_TOPOLOGIES,
                        choices=ALL_TOPOLOGIES,
                        help="Topologies to run (default: all 7)")
    parser.add_argument("--scales",      nargs="+", type=int, default=ALL_SCALES,
                        help="Agent counts N (default: 8 16 32 64 128)")
    parser.add_argument("--seeds",       nargs="+", type=int, default=ALL_SEEDS,
                        help="Random seeds (default: 0 1 2)")
    parser.add_argument("--model",       default="gpt-4o-mini",
                        help="LLM model name (default: gpt-4o-mini)")
    parser.add_argument("--data-root",   default="data/sweep",
                        help="Root output directory (default: data/sweep)")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Parallel process workers (default: 1 = sequential)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip runs that already have run_metadata.json")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print plan only, no LLM calls")
    args = parser.parse_args()

    # ── Deduplicate benchmarks list ──────────────────────────────────────────
    seen_benchmarks = []
    for b in args.benchmarks:
        if b not in seen_benchmarks:
            seen_benchmarks.append(b)
    args.benchmarks = seen_benchmarks

    sweep_dir = ROOT / args.data_root
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sweep_dir / "sweep_manifest.jsonl"
    errors_path   = sweep_dir / "sweep_errors.jsonl"

    # ── Header ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  MAS Power-Law Sweep")
    print("="*65)
    print(f"  Model:       {args.model}")
    print(f"  Benchmarks:  {args.benchmarks}")
    print(f"  Topologies:  {args.topologies}")
    print(f"  Scales (N):  {args.scales}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Workers:     {args.workers}")
    print(f"  Resume:      {args.resume}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Output:      {sweep_dir}")

    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("\nERROR: OPENAI_API_KEY not set.")
            print("       Run: export $(grep -v '^#' .env | xargs)")
            sys.exit(1)
        print(f"  API key:     {api_key[:12]}...")

    # ── Build grid ────────────────────────────────────────────────────────────
    specs, _ = build_run_specs(
        benchmarks=args.benchmarks,
        topologies=args.topologies,
        scales=args.scales,
        seeds=args.seeds,
        sweep_dir=sweep_dir,
        resume=args.resume,
    )

    skipped = 0
    if args.resume:
        # Count skipped for display
        all_specs, _ = build_run_specs(
            benchmarks=args.benchmarks,
            topologies=args.topologies,
            scales=args.scales,
            seeds=args.seeds,
            sweep_dir=sweep_dir,
            resume=False,
        )
        skipped = len(all_specs) - len(specs)

    total_planned = len(specs) + skipped
    print(f"\n  Planned:     {total_planned} runs total")
    if args.resume and skipped:
        print(f"  Skipping:    {skipped} already completed")
    print(f"  Queued:      {len(specs)} runs to execute")

    if args.dry_run:
        print("\n  DRY RUN — first 10 specs:")
        for s in specs[:10]:
            print(f"    {s.key()}")
        if len(specs) > 10:
            print(f"    ... and {len(specs)-10} more")
        print()
        sys.exit(0)

    if not specs:
        print("\n  Nothing to run. Use --resume=false to re-run existing.")
        sys.exit(0)

    # ── Write sweep plan ──────────────────────────────────────────────────────
    plan = {
        "model":      args.model,
        "benchmarks": args.benchmarks,
        "topologies": args.topologies,
        "scales":     args.scales,
        "seeds":      args.seeds,
        "total_runs": total_planned,
        "queued":     len(specs),
        "skipped":    skipped,
        "started":    _now(),
        "sweep_dir":  str(sweep_dir),
    }
    (sweep_dir / "sweep_plan.json").write_text(json.dumps(plan, indent=2))

    # ── Execute ───────────────────────────────────────────────────────────────
    progress = Progress(len(specs), sweep_dir)
    t_start  = time.time()

    print(f"\n  Starting sweep at {_now()}")
    print(f"  {'─'*60}")

    def _write_result(result: Dict):
        line = json.dumps(result) + "\n"
        with open(manifest_path, "a") as f:
            f.write(line)
        if result.get("status") not in ("ok", "dry_run", "skipped"):
            with open(errors_path, "a") as f:
                f.write(line)

    if args.workers == 1:
        # Sequential — simpler, better for debugging and rate-limit control
        for spec in specs:
            result  = _run_one(spec, args.model, dry_run=False)
            status  = result.get("status", "error")
            extra   = ""
            if status == "ok":
                extra = f"events={result.get('event_count',0)}  score={result.get('score')}"
            elif result.get("error"):
                extra = result["error"].splitlines()[-1][:60]
            progress.update(status)
            progress.print_line(spec, status, extra)
            _write_result(result)

    else:
        # Parallel — ProcessPoolExecutor, one process per worker
        # Each worker loads its own LLM client (thread-safe, rate-limited by OpenAI)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, spec, args.model, False): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec    = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "key":       spec.key(),
                        "status":    "fatal",
                        "run_dir":   str(spec.run_dir),
                        "error":     traceback.format_exc()[:500],
                        "timestamp": _now(),
                    }
                status  = result.get("status", "error")
                extra   = ""
                if status == "ok":
                    extra = f"events={result.get('event_count',0)}  score={result.get('score')}"
                elif result.get("error"):
                    extra = result["error"].splitlines()[-1][:60]
                progress.update(status)
                progress.print_line(spec, status, extra)
                _write_result(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed  = time.time() - t_start
    elapsed_h = elapsed / 3600

    print(f"\n  {'─'*60}")
    print(f"  Sweep complete at {_now()}")
    print(f"  Elapsed:  {elapsed_h:.2f}h  ({elapsed:.0f}s)")
    print(f"  Total:    {progress.done} runs")
    print(f"  OK:       {progress.ok}")
    print(f"  Errors:   {progress.errors}")
    print(f"  Skipped:  {progress.skipped}")
    print(f"\n  Manifest: {manifest_path}")
    print(f"  Errors:   {errors_path}")
    print(f"  Data:     {sweep_dir}")

    if progress.errors > 0:
        print(f"\n  {progress.errors} runs had errors — check sweep_errors.jsonl")

    # Write final progress
    progress._write()

    # Write completion marker
    (sweep_dir / "sweep_complete.json").write_text(json.dumps({
        "completed":  _now(),
        "total_runs": progress.done,
        "ok":         progress.ok,
        "errors":     progress.errors,
        "skipped":    progress.skipped,
        "elapsed_s":  round(elapsed),
    }, indent=2))

    print()


if __name__ == "__main__":
    main()
