# claim-traces/

Example raw interaction traces and pooled analysis output from the
experimental sweep, illustrating the data formats the pipeline produces
and consumes.

This is a **small representative sample**, not the full experimental
dataset. The complete trace set (all benchmarks × topologies × scales ×
seeds) is ~2.7GB across ~2,000 runs and is not checked into this
repository; only the subset below is tracked in git.

## What's included

**Same task, seven topologies** (`GAIA/*/n32/s0/04a04a9b-.../`) — the
identical GAIA task run under all seven communication topologies at
N=32, so the raw traces are directly comparable across topology.

**One additional run per other benchmark** (`MARBLE/`, `REALM/`,
`SWE-bench/`, each `star/n32/s0/...`) — for format diversity across
benchmark suites.

Each run directory contains:
- `events.jsonl` — the raw per-turn TraceRow log (one JSON object per
  coordination event: claim lineage, subtask lineage, reasoning text,
  coordination signals). This is the direct input to
  `src/event_extraction/` and everything downstream.
- `snapshots.jsonl` — periodic graph-structure snapshots (agent
  activation, edge weights) taken during the run.
- `run_config.json` — the resolved configuration for this run.
- `run_metadata.json` — post-run summary statistics (task success,
  claim/subtask counts, token/latency totals).

**`combined/`** — pooled analysis output across the *full* sweep (not
just this sample): the reconstructed claim DAG (`claim_nodes.csv`,
`claim_edges.csv.gz`, `agent_edges.csv.gz`), a per-run graph summary
(`run_graph_summary.csv`), and the fitted tail statistics
(`analysis/fit_results.json`, produced by
`src/tail_fitting/powerlaw_fit.py`). This is small enough to include in
full and is the actual data behind the paper's reported statistics.

## Regenerating your own traces

See the root [README](../README.md#quick-start) for how to produce
traces in this exact format via `scripts/run_sweep.py` /
`scripts/main.py`, and `scripts/extract_events.py` /
`scripts/run_analysis.py` for how `combined/` is produced from raw traces.
