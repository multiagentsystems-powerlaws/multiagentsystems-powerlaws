"""
execution/runner.py
--------------------
Async sweep runner.  Runs the full combinatorial grid:

    benchmarks × topologies × agent_scales × seeds

Concurrency is controlled via an asyncio.Semaphore so you don't
blow up the OpenAI rate limit on Tinkercliffs.

Features
--------
- Resume:  skips runs where run_metadata.json already exists
- Progress: tqdm bar + per-run stdout logging
- Failures: caught per-run; written to data/failed_runs.jsonl
- Manifest: final summary written to data/sweep_manifest.json

Usage (programmatic)
---------------------
    runner = SweepRunner(llm=llm, data_root=Path("data/runs"))
    results = await runner.run_sweep(
        tasks=tasks,
        topologies=[TopologyName.CHAIN, TopologyName.STAR, ...],
        agent_scales=[4, 8, 16, 32],
        seeds=[0, 1, 2],
        max_concurrent=8,
    )

Usage (CLI)
-----------
    python scripts/run_sweep.py --benchmark MARBLE --topology star --n 8 --seed 0
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loggers.schemas import TopologyName, RoutingStrategy, MemoryType
from execution.graph_runner import BenchmarkTask, GraphRunner, RunResult


# ────────────────────────────────────────────────────────────────
# Sweep configuration
# ────────────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """Parameters for one full sweep."""
    topologies:    List[TopologyName]
    agent_scales:  List[int]           # e.g. [8, 16, 32, 64, 128]
    seeds:         List[int]           # e.g. [0, 1, 2]
    max_concurrent: int = 8            # simultaneous runs
    skip_existing:  bool = True        # resume by skipping done runs
    architecture_override: str = ""
    snapshot_every: int = 5
    max_steps:      int = 50
    model_name:     str = "gpt-4o-mini"
    temperature:    float = 0.7


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

class SweepRunner:
    """
    Async runner for the full combinatorial sweep.

    Parameters
    ----------
    llm         LangChain chat model (must support async if max_concurrent > 1).
    data_root   Root directory for all run artefacts.
    """

    def __init__(
        self,
        *,
        llm,
        data_root: Path = Path("data/runs"),
    ) -> None:
        self.llm       = llm
        self.data_root = Path(data_root)

    # ── Public API ──────────────────────────────────────────────────

    async def run_sweep(
        self,
        tasks:  List[BenchmarkTask],
        config: SweepConfig,
    ) -> List[RunResult]:
        """
        Run all (task × topology × scale × seed) combinations.
        Returns list of RunResult (one per completed run).
        """
        # Build the full job list
        jobs = []
        for task in tasks:
            for topo in config.topologies:
                for n in config.agent_scales:
                    for seed in config.seeds:
                        jobs.append((task, topo, n, seed))

        total = len(jobs)
        print(f"\nSweep: {total} runs  "
              f"({len(tasks)} tasks × {len(config.topologies)} topologies × "
              f"{len(config.agent_scales)} scales × {len(config.seeds)} seeds)")
        print(f"Max concurrent: {config.max_concurrent}  |  Resume: {config.skip_existing}\n")

        sem           = asyncio.Semaphore(config.max_concurrent)
        attempt_delays: dict = {}  # run_id -> cumulative backoff seconds
        results: List[RunResult] = []
        failed:  List[Dict]      = []
        done    = 0

        async def run_one(job):
            nonlocal done
            task, topo, n, seed = job
            run_id = (
                f"{task.benchmark}__{topo.value}__n{n}__s{seed}__{task.task_id}"
            )
            # Layout: {data_root}/{topology}/n{N}/s{seed}/{task_id}/
            # data_root is already benchmark-specific when using --data-root per GPU.
            run_dir = (
                self.data_root / topo.value
                / f"n{n}" / f"s{seed}" / task.task_id
            )

            # Resume check
            if config.skip_existing and (run_dir / "run_metadata.json").exists():
                done += 1
                return None

            async with sem:
                runner = GraphRunner(
                    llm=self.llm,
                    data_root=self.data_root,
                    architecture=config.architecture_override,
                    snapshot_every=config.snapshot_every,
                    max_steps=config.max_steps,
                    model_name=config.model_name,
                    temperature=config.temperature,
                )
                t0 = time.time()
                try:
                    # Run in thread pool (GraphRunner.run is sync)
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: runner.run(task, topo, n, seed, run_id=run_id),
                    )
                    results.append(result)
                    done += 1
                    elapsed = time.time() - t0
                    status  = "OK" if result.success else "FAIL"
                    print(
                        f"[{done:4d}/{total}] {status}  "
                        f"{task.benchmark}/{topo.value}/n{n}/s{seed}/{task.task_id}  "
                        f"{elapsed:.1f}s  events={result.event_count}  "
                        f"tokens={result.tokens_total}"
                    )
                    return result

                except Exception:
                    tb  = traceback.format_exc()
                    err = tb.lower()
                    # Rate-limit: pause so other concurrent jobs can drain
                    if any(x in err for x in ["429", "rate limit", "ratelimit", "too many requests"]):
                        wait = 30 + attempt_delays.get(run_id, 0)
                        attempt_delays[run_id] = wait  # escalate on repeated hits
                        print(f"[{done:4d}/{total}] RATE_LIMIT {run_id} — sleeping {wait}s")
                        await asyncio.sleep(wait)
                    failed.append({
                        "run_id": run_id, "error": tb,
                        "task_id": task.task_id, "topology": topo.value,
                        "n": n, "seed": seed,
                    })
                    done += 1
                    print(f"[{done:4d}/{total}] ERROR  {run_id}\n  {tb.splitlines()[-1]}")
                    return None

        # Launch all jobs
        await asyncio.gather(*[run_one(j) for j in jobs])

        # Write failed runs log
        if failed:
            fail_path = self.data_root.parent / "failed_runs.jsonl"
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_path, "a") as f:
                for entry in failed:
                    f.write(json.dumps(entry) + "\n")
            print(f"\n{len(failed)} failed runs logged → {fail_path}")

        # Write sweep manifest
        self._write_manifest(results, config)

        print(f"\nSweep complete: {len(results)} successful, {len(failed)} failed.")
        return results

    # ── Manifest ────────────────────────────────────────────────────

    def _write_manifest(
        self,
        results: List[RunResult],
        config:  SweepConfig,
    ) -> None:
        manifest = {
            "total_runs":      len(results),
            "topologies":      [t.value for t in config.topologies],
            "agent_scales":    config.agent_scales,
            "seeds":           config.seeds,
            "benchmarks":      sorted({r.benchmark for r in results}),
            "total_events":    sum(r.event_count for r in results),
            "total_tokens":    sum(r.tokens_total for r in results),
            "total_wall_time": round(sum(r.wall_time_s for r in results), 1),
            "runs": [
                {
                    "run_id":       r.run_id,
                    "benchmark":    r.benchmark,
                    "topology":     r.topology,
                    "num_agents":   r.num_agents,
                    "seed":         r.seed,
                    "task_id":      r.task_id,
                    "success":      r.success,
                    "score":        r.score,
                    "events":       r.event_count,
                    "tokens":       r.tokens_total,
                    "wall_time_s":  r.wall_time_s,
                }
                for r in results
            ],
        }
        out = self.data_root.parent / "sweep_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2))
        print(f"Manifest → {out}")