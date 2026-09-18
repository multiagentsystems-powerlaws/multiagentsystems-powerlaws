"""
scripts/test_task_expander.py
=============================
Integration test for src/benchmark_wrappers/task_expander.py.

Uses your REAL benchmark wrappers — no mock data.

Run from the repo root:

    cd ~/mas-powerlaws

    # Synthetic expansion (no API) — tests tree structure with real task text
    python scripts/test_task_expander.py --benchmark gaia --N 32
    python scripts/test_task_expander.py --benchmark realm --N 64
    python scripts/test_task_expander.py --benchmark swebench --N 32
    python scripts/test_task_expander.py --benchmark marble --N 64

    # Full suite across all benchmarks and N values (synthetic, no API)
    python scripts/test_task_expander.py --all

    # Real LLM expansion — sees actual Claude-generated sub-tasks
    python scripts/test_task_expander.py --mode llm --benchmark gaia --N 32

Repo layout:
    ~/mas-powerlaws/
        src/
            benchmark_wrappers/
                task_expander.py
                gaia.py
                realm_bench.py
                swebench.py
                marble.py
        scripts/
            test_task_expander.py   <-- this file
"""

import argparse
import sys
from collections import deque
from pathlib import Path

# ── sys.path so imports resolve from src/ ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

# ── task_expander ─────────────────────────────────────────────────────────────
try:
    from benchmark_wrappers.task_expander import (
        TaskExpander,
        validate_tree,
        evaluate_accuracy,
        save_tree,
        load_tree_dict,
        num_seed_tasks,
        num_expanded_per_seed,
        total_pool_size,
        agents_per_task,
    )
except ModuleNotFoundError as e:
    print(f"\nImport failed: {e}")
    print(f"Expected: {SRC_ROOT / 'benchmark_wrappers' / 'task_expander.py'}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Load real tasks from your existing benchmark wrappers
# Each returns a list of BenchmarkTask objects with .task_id, .prompt,
# .gold_answer, .task_family, .difficulty, .benchmark attributes.
# We convert to the dict format task_expander expects.
# ---------------------------------------------------------------------------

def _to_expander_fmt(tasks) -> list[dict]:
    """Convert BenchmarkTask objects to the dict format task_expander expects."""
    return [
        {
            "node_id":          t.task_id,
            "description":      t.prompt,
            "ground_truth":     t.gold_answer,
            "benchmark_source": t.benchmark,
        }
        for t in tasks
    ]


def load_pool(benchmark: str) -> tuple[list[dict], str]:
    """
    Load a real benchmark pool using your existing wrappers.
    Returns (pool_as_dicts, domain_string).
    """
    if benchmark == "gaia":
        from benchmark_wrappers.gaia import load_gaia_tasks
        tasks = load_gaia_tasks(max_tasks=150)
        return _to_expander_fmt(tasks), "qa"

    elif benchmark == "realm":
        from benchmark_wrappers.realm_bench import load_realm_tasks
        datasets_path = REPO_ROOT / "REALM-Bench" / "datasets"
        tasks = load_realm_tasks(max_tasks=150, datasets_path=datasets_path)
        return _to_expander_fmt(tasks), "planning"

    elif benchmark == "swebench":
        from benchmark_wrappers.swebench import load_swebench_tasks
        tasks = load_swebench_tasks(max_tasks=235)
        return _to_expander_fmt(tasks), "coding"

    elif benchmark == "marble":
        from benchmark_wrappers.marble import load_marble_tasks
        tasks = load_marble_tasks(max_tasks=150)
        return _to_expander_fmt(tasks), "reasoning"

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_scaling_table():
    section("Scaling table — paper Section H (K=min(5,|B|), M=ceil(N/K*5))")
    print(f"  {'N':>5}  {'K':>4}  {'M':>5}  {'K*M':>6}  {'agents/task':>12}")
    print(f"  {'-'*42}")
    for N in [8, 16, 32, 64, 128, 256, 512]:
        K    = num_seed_tasks(150)
        M    = num_expanded_per_seed(N, K)
        pool = total_pool_size(N, K)
        apt  = agents_per_task(N, K)
        print(f"  {N:>5}  {K:>4}  {M:>5}  {pool:>6}  {apt:>12.1f}")


def print_tree(tree, max_exp: int = 3):
    section("Tree structure (first 3 expanded tasks per seed shown)")
    desc = tree.root_node.description
    print(f"\n[ROOT]  {tree.root_node.node_id}")
    print(f"  desc   : \"{desc[:88]}{'...' if len(desc)>88 else ''}\"")
    print(f"  agents : {tree.root_node.agent_budget}")
    print(f"  deps   : {tree.root_node.depends_on}")

    for seed_n in tree.seed_nodes:
        print(f"\n  [SEED]  {seed_n.node_id}  (agents={seed_n.agent_budget})")
        # Show the real task text — this is the point of using real benchmarks
        print(f"    desc : \"{seed_n.description[:100]}\"")
        if seed_n.ground_truth:
            print(f"    gt   : \"{str(seed_n.ground_truth)[:80]}\"")

        children = [n for n in tree.expanded_nodes
                    if n.seed_parent_id == seed_n.node_id]
        for exp_n in children[:max_exp]:
            deps = f"  <- {exp_n.depends_on}" if exp_n.depends_on else ""
            print(f"    [EXP]  {exp_n.node_id}  (agents={exp_n.agent_budget}){deps}")
            # Show description — in LLM mode this is real Claude-generated text
            print(f"           \"{exp_n.description[:90]}\"")
        if len(children) > max_exp:
            print(f"    ... +{len(children)-max_exp} more expanded tasks")


def print_dag_stats(tree):
    section("DAG statistics")
    all_nodes   = tree.all_nodes
    total_edges = sum(len(n.depends_on) for n in all_nodes)
    cross = sum(
        1 for n in tree.expanded_nodes
        for dep in n.depends_on
        if (dn := tree.node_by_id(dep)) and
           dn.node_type == "expanded" and
           dn.seed_parent_id != n.seed_parent_id
    )
    within = total_edges - cross - len(tree.seed_nodes)
    print(f"  Total DAG edges      : {total_edges}")
    print(f"  Within-cluster       : {within}")
    print(f"  Cross-cluster        : {cross}")
    print(f"  Seed -> root         : {len(tree.seed_nodes)}")
    print(f"  Nodes with any dep   : "
          f"{sum(1 for n in all_nodes if n.depends_on)} / {len(all_nodes)}")


def print_allocation(tree):
    section("Agent allocation")
    budgets = [tree.agent_allocation.get(n.node_id, 0) for n in tree.execution_pool]
    total   = sum(tree.agent_allocation.values())
    print(f"  N (requested)        : {tree.N}")
    print(f"  Allocated total      : {total}  ({'OK' if total==tree.N else 'MISMATCH'})")
    print(f"  Root budget          : {tree.root_node.agent_budget}")
    print(f"  Pool budgets         : min={min(budgets)}  max={max(budgets)}  "
          f"avg={sum(budgets)/len(budgets):.2f}")
    print(f"  Active agent frac    : {tree.active_agent_fraction():.1%}  "
          f"(paper target >80%)")


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def run_checks(tree, pool_size: int, N: int) -> list[str]:
    fails = []

    # 1. Agent allocation sums to exactly N
    total = sum(tree.agent_allocation.values())
    if total != N:
        fails.append(f"Agent sum {total} != N={N}")

    # 2. Root depends on exactly the K seed nodes
    exp_root = {n.node_id for n in tree.seed_nodes}
    got_root = set(tree.root_node.depends_on)
    if got_root != exp_root:
        fails.append(f"Root deps wrong — missing:{exp_root-got_root}  extra:{got_root-exp_root}")

    # 3. Each seed has exactly M expanded children
    for seed_n in tree.seed_nodes:
        ch = [n for n in tree.expanded_nodes if n.seed_parent_id == seed_n.node_id]
        if len(ch) != tree.M:
            fails.append(f"{seed_n.node_id}: {len(ch)} children != M={tree.M}")

    # 4. No self-loops
    for n in tree.all_nodes:
        if n.node_id in n.depends_on:
            fails.append(f"Self-loop: {n.node_id}")

    # 5. All dependency references point to real node_ids
    valid = {n.node_id for n in tree.all_nodes}
    for n in tree.all_nodes:
        bad = [d for d in n.depends_on if d not in valid]
        if bad:
            fails.append(f"{n.node_id} has dangling deps: {bad}")

    # 6. K and M match paper formulas exactly
    exp_K = num_seed_tasks(pool_size)
    exp_M = num_expanded_per_seed(N, exp_K)
    if tree.K != exp_K:
        fails.append(f"K={tree.K} but formula gives {exp_K}")
    if tree.M != exp_M:
        fails.append(f"M={tree.M} but formula gives {exp_M}")

    # 7. Full DAG acyclicity via Kahn's algorithm
    # (list-position check is wrong for cross-cluster edges — must do real topo sort)
    in_degree = {n.node_id: 0 for n in tree.all_nodes}
    fwd: dict = {n.node_id: [] for n in tree.all_nodes}
    for n in tree.all_nodes:
        for dep in n.depends_on:
            if dep in in_degree:
                in_degree[n.node_id] += 1
            if dep in fwd:
                fwd[dep].append(n.node_id)
    queue   = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        nid = queue.popleft()
        visited += 1
        for child in fwd.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if visited != len(tree.all_nodes):
        cycle_nodes = [nid for nid, deg in in_degree.items() if deg > 0]
        fails.append(f"DAG has cycle(s) involving nodes: {cycle_nodes[:5]}")

    # 8. Every expanded node has a valid seed_parent_id
    seed_ids = {n.node_id for n in tree.seed_nodes}
    for n in tree.expanded_nodes:
        if n.seed_parent_id not in seed_ids:
            fails.append(f"{n.node_id} bad seed_parent_id={n.seed_parent_id}")

    return fails


# ---------------------------------------------------------------------------
# Single benchmark test (verbose)
# ---------------------------------------------------------------------------

def test_single(benchmark: str, N: int, use_llm: bool, seed: int = 42) -> bool:
    print(f"\nLoading tasks from {benchmark} wrapper...")
    try:
        pool, domain = load_pool(benchmark)
    except Exception as e:
        print(f"  ERROR loading {benchmark}: {e}")
        print("  Check that the benchmark wrapper and its data source are available.")
        return False

    print(f"  Loaded {len(pool)} tasks from {benchmark}")
    mode = "LLM" if use_llm else "synthetic"
    print(f"\nBuilding tree | benchmark={benchmark}  domain={domain}  "
          f"N={N}  mode={mode}  seed={seed}")

    try:
        exp  = TaskExpander(benchmark=benchmark, domain=domain, seed=seed)
        tree = exp.build(N=N, benchmark_pool=pool, use_llm=use_llm)
    except Exception as e:
        print(f"  ERROR building tree: {e}")
        return False

    print(tree.summary())
    print_tree(tree)
    print_dag_stats(tree)
    print_allocation(tree)

    section("Invariant checks")
    fails = run_checks(tree, len(pool), N)
    if fails:
        print("  FAILURES:")
        for f in fails:
            print(f"    x  {f}")
    else:
        print("  All checks passed.")

    # evaluate_accuracy — only meaningful when seeds have ground truth
    seeds_with_gt = [n for n in tree.seed_nodes if n.ground_truth]
    if seeds_with_gt:
        for i, n in enumerate(tree.seed_nodes):
            n.agent_answer = n.ground_truth if (i % 2 == 0 and n.ground_truth) else "wrong"
        acc     = evaluate_accuracy(tree)
        exp_acc = sum(1 for i, n in enumerate(tree.seed_nodes)
                      if i % 2 == 0 and n.ground_truth) / tree.K
        acc_ok  = acc is not None and abs(acc - exp_acc) < 0.02
        print(f"\n  evaluate_accuracy  : {acc:.3f}  (expected ~{exp_acc:.3f})  "
              f"{'OK' if acc_ok else 'FAIL'}")
    else:
        acc_ok = True
        print(f"\n  evaluate_accuracy  : skipped (no ground truth in {benchmark})")

    # Serialisation round-trip
    out = REPO_ROOT / "data" / f"test_tree_{benchmark}_N{N}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tree(tree, str(out))
    d     = load_tree_dict(str(out))
    rt_ok = (d["K"]==tree.K and d["N"]==tree.N and
             d["M"]==tree.M and len(d["nodes"])==tree.total_nodes)
    print(f"  serialisation      : {'OK' if rt_ok else 'FAIL'}  -> {out}")

    # evaluate_accuracy returns None before any answers
    tree2   = TaskExpander(benchmark, domain, seed).build(
                  N=N, benchmark_pool=pool, use_llm=False)
    none_ok = evaluate_accuracy(tree2) is None
    print(f"  accuracy=None (unanswered): {'OK' if none_ok else 'FAIL'}")

    overall = len(fails) == 0 and acc_ok and rt_ok and none_ok
    print(f"\n  Result: {'PASS' if overall else 'FAIL'}")
    return overall


# ---------------------------------------------------------------------------
# Full suite — all available benchmarks x all N values
# ---------------------------------------------------------------------------

def test_all(benchmarks: list[str]) -> bool:
    section(f"Full synthetic suite — {len(benchmarks)} benchmarks x 7 N values")

    # Load all pools first so we fail fast on missing wrappers
    pools: dict[str, tuple[list, str]] = {}
    for bench in benchmarks:
        print(f"Loading {bench}...", end=" ", flush=True)
        try:
            pool, domain = load_pool(bench)
            pools[bench] = (pool, domain)
            print(f"{len(pool)} tasks")
        except Exception as e:
            print(f"FAILED: {e}")

    if not pools:
        print("No benchmarks loaded. Aborting.")
        return False

    print(f"\n  {'benchmark':20s}  {'N':>4}  {'K':>2}  {'M':>3}  {'pool':>5}  "
          f"{'result':>6}  {'frac':>6}  {'depth':>5}")
    print(f"  {'-'*62}")

    all_ok = True
    for bench, (pool, domain) in pools.items():
        for N in [8, 16, 32, 64, 128, 256, 512]:
            try:
                exp   = TaskExpander(benchmark=bench, domain=domain, seed=42)
                tree  = exp.build(N=N, benchmark_pool=pool, use_llm=False)
                m     = validate_tree(tree)
                fails = run_checks(tree, len(pool), N)
                ok    = len(fails) == 0
            except Exception as e:
                fails = [str(e)]
                ok    = False
                m     = {"K":0,"M":0,"pool_size":0,"active_agent_fraction":0,"max_dag_depth":0}

            if not ok:
                all_ok = False
            print(f"  {bench:20s}  {N:>4}  {m['K']:>2}  {m['M']:>3}  "
                  f"{m['pool_size']:>5}  {'PASS' if ok else 'FAIL':>6}  "
                  f"{m['active_agent_fraction']:>5.0%}  {m['max_dag_depth']:>5}")
            for f in fails:
                print(f"      x  {f}")

    print(f"\n  {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

BENCHMARKS = ["gaia", "realm", "swebench", "marble"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Integration test for src/benchmark_wrappers/task_expander.py"
    )
    ap.add_argument("--mode",      choices=["synthetic", "llm"], default="synthetic",
                    help="synthetic=no API calls, llm=real Claude expansion")
    ap.add_argument("--N",         type=int, default=32)
    ap.add_argument("--benchmark", choices=BENCHMARKS, default="gaia")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--all",       action="store_true",
                    help="Run full suite across all benchmarks and all N values")
    args = ap.parse_args()

    print_scaling_table()

    if args.all:
        ok = test_all(BENCHMARKS)
        sys.exit(0 if ok else 1)

    use_llm = args.mode == "llm"
    if use_llm:
        import os
        from dotenv import load_dotenv
        load_dotenv()  # picks up OPENAI_API_KEY from .env
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nERROR: set OPENAI_API_KEY before running --mode llm")
            sys.exit(1)

    ok = test_single(args.benchmark, args.N, use_llm, args.seed)
    sys.exit(0 if ok else 1)