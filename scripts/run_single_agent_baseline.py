#!/usr/bin/env python3
"""
scripts/run_single_agent_baseline.py
--------------------------------------
Runs the same task portfolio with a single agent (no MAS).
Produces Table 2 comparison data.

The single-agent baseline uses the same LLM, same tasks, same prompt
structure as the MAS runs — but with only one agent and no topology.
Output goes to a separate data-root so it never pollutes MAS data.

Usage
-----
  python scripts/run_single_agent_baseline.py \\
      --benchmarks MARBLE GAIA REALM SWE-bench \\
      --data-root /scratch/kavanav/mas-results/single_agent \\
      --model gpt-4o-mini \\
      --seeds 0 1 2

Slurm
-----
  sbatch slurm/sweep_single_agent.slurm
"""
import argparse, asyncio, json, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loggers.event_bus import EventBus
from loggers.schemas import RunConfig, RunMetadata, TopologyName, RoutingStrategy, MemoryType
from execution.graph_runner import BenchmarkTask
from prompts.templates import build_system_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", nargs="+",
                   choices=["MARBLE","GAIA","REALM","SWE-bench"],
                   default=["MARBLE"])
    p.add_argument("--data-root", default="data/single_agent")
    p.add_argument("--model",     default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seeds",     type=int, nargs="+", default=[0,1,2])
    p.add_argument("--max-tasks", type=int, default=20)
    p.add_argument("--resume",    action="store_true")
    return p.parse_args()


from llm_factory import build_llm


def run_single_agent(
    llm, task: BenchmarkTask, seed: int, data_root: Path, resume: bool
) -> dict:
    """Run one task with a single agent, log events, return stats."""
    run_id  = f"single__{task.benchmark}__s{seed}__{task.task_id}"
    run_dir = data_root / task.benchmark / f"s{seed}" / task.task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if resume and (run_dir / "run_metadata.json").exists():
        return {"skipped": True, "run_id": run_id}

    bus = EventBus(run_dir)
    bus.write_run_config(RunConfig(
        run_id=run_id, benchmark=task.benchmark,
        task_id=task.task_id, task_family=task.task_family,
        difficulty=task.difficulty, topology=TopologyName.CHAIN,
        architecture="single_agent",
        routing_strategy=RoutingStrategy.PLANNER_ASSIGNED,
        memory_type=MemoryType.FULL_HISTORY,
        num_agents=1, max_steps=1, seed=seed,
        run_seed=seed, task_seed=seed, topology_seed=seed,
    ))

    system_prompt = build_system_prompt(
        topology="chain", task_family=task.task_family,
        agent_id="agent_000", agent_index=0, num_agents=1,
        round=1, max_rounds=1, depth=0, role="solver",
        supervisor_id="none", worker_ids="[]", neighbor_ids="[]",
        community_id=0, community_member_ids="[]",
        bridge_id="none", num_consulted=0, step=1, max_steps=1,
        role_block="You are a single AI agent.",
        dispatch_or_work_block="Solve the task independently.",
    )

    from loggers.schemas import AgentEvent, EventType, ClaimType, ClaimStatus
    import hashlib, uuid, time as _time

    t0 = _time.time()
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"TASK: {task.prompt}"},
        ]
        resp   = llm.invoke(messages)
        answer = resp.content if hasattr(resp, "content") else str(resp)
        success = True
    except Exception as e:
        answer  = f"[ERROR: {e}]"
        success = False

    elapsed = _time.time() - t0
    in_tok  = len(system_prompt + task.prompt) // 4
    out_tok = len(answer) // 4

    bus.log_event(AgentEvent(
        run_id=run_id, benchmark=task.benchmark,
        task_id=task.task_id, task_family=task.task_family,
        difficulty=task.difficulty, topology=TopologyName.CHAIN,
        architecture="single_agent", num_agents=1, seed=seed,
        step_id=1, agent_id="agent_000", agent_role="solver",
        event_type=EventType.FINALIZE_ANSWER,
        message_id=f"msg_{uuid.uuid4().hex[:8]}",
        claim_id=f"claim_{uuid.uuid4().hex[:8]}",
        claim_type=ClaimType.FINAL_CLAIM,
        claim_status=ClaimStatus.ACCEPTED,
        claim_depth=0,
        claim_text_hash=hashlib.md5(answer.encode()).hexdigest()[:12],
        tokens_input=in_tok, tokens_output=out_tok,
        tokens_total_event=in_tok + out_tok,
        latency_ms=round(elapsed * 1000, 2),
        action_success=success,
    ))

    # Simple token-overlap score
    score = 0.0
    if task.gold_answer and answer:
        a_toks = set(answer.lower().split())
        g_toks = set(task.gold_answer.lower().split())
        if g_toks:
            overlap = len(a_toks & g_toks)
            p = overlap / len(a_toks) if a_toks else 0
            r = overlap / len(g_toks)
            score = round(2*p*r/(p+r), 4) if (p+r) > 0 else 0.0

    bus.flush_run_outcome(RunMetadata(
        run_id=run_id, task_success=success and score > 0,
        task_score=score, tokens_total=in_tok + out_tok,
        messages_total=1, num_unique_agents_activated=1,
        num_coordination_events_total=1,
        wall_time_seconds=round(elapsed, 2),
    ))
    bus.close()

    return {"run_id": run_id, "score": score, "tokens": in_tok + out_tok,
            "elapsed": elapsed, "success": success}


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    print(f"\nSingle-agent baseline")
    print(f"  Benchmarks: {args.benchmarks}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Data root:  {data_root}")

    llm = build_llm(args.model, args.temperature)

    # Load tasks
    all_tasks = []
    for bench in args.benchmarks:
        try:
            if bench == "MARBLE":
                from benchmark_wrappers.marble import load_marble_tasks
                all_tasks += load_marble_tasks(args.max_tasks)
            elif bench == "GAIA":
                from benchmark_wrappers.gaia import load_gaia_tasks
                all_tasks += load_gaia_tasks(args.max_tasks)
            elif bench == "REALM":
                from benchmark_wrappers.realm_bench import load_realm_tasks
                all_tasks += load_realm_tasks(args.max_tasks)
            elif bench == "SWE-bench":
                from benchmark_wrappers.swebench import load_swebench_tasks
                all_tasks += load_swebench_tasks(min(args.max_tasks, 10))
            print(f"  {bench}: loaded tasks")
        except Exception as e:
            print(f"  {bench}: FAILED — {e}")

    total = len(all_tasks) * len(args.seeds)
    done  = 0
    print(f"\nRunning {total} single-agent evaluations...\n")

    for task in all_tasks:
        for seed in args.seeds:
            result = run_single_agent(llm, task, seed, data_root, args.resume)
            done  += 1
            if result.get("skipped"):
                print(f"  [{done:3d}/{total}] SKIP  {task.task_id}")
            else:
                print(f"  [{done:3d}/{total}] OK    {task.task_id}  "
                      f"score={result['score']:.3f}  "
                      f"tokens={result['tokens']}  "
                      f"t={result['elapsed']:.1f}s")

    print(f"\nBaseline complete. Results → {data_root}/")
    print("Run Table 2: python scripts/generate_tables.py "
          f"--single-agent-root {data_root}")


if __name__ == "__main__":
    main()
