================================================================================
MAS POWER-LAWS FULL SWEEP — TINKERCLIFFS2
COLM 2026  |  4,900 runs
================================================================================

SWEEP SPEC
  Topologies:  7 (chain, star, tree, full_mesh, sparse_mesh, hybrid_modular, dynamic_reputation)
               NOTE: paper's reported results use only 6 of these (hybrid_modular
               has no corresponding description anywhere in the paper) — decide
               whether to include it before treating this as the paper-aligned sweep.
  Benchmarks:  4 (MARBLE, GAIA, REALM, SWE-bench)
               NOTE: MARBLE stands in for the paper's "MultiAgentBench" — not the
               same benchmark; the real MultiAgentBench data is unused in MARBLE/multiagentbench/.
  Tasks:       5 per benchmark
  Seeds:       5 (0, 1, 2, 3, 4)
  Scales:      7 (N=8, 16, 32, 64, 128, 256, 512)
  Total runs:  4,900
  Est. tokens: ~194M
  Est. cost:   ~$44 (gpt-4o-mini @ $0.15/1M)
  Est. time:   ~52h total wallclock (4 GPUs running in parallel, max_concurrent=4)

STRATEGY: Run one benchmark per GPU node simultaneously.
Each GPU runs all 7 topologies × 5 tasks × 5 seeds × 7 scales = 1,225 runs.

================================================================================
STEP 0 — ENVIRONMENT SETUP (run once on login node)
================================================================================

cd ~/mas-powerlaws
source venv/bin/activate
echo "export OPENAI_API_KEY=sk-..." >> ~/.bashrc
source ~/.bashrc

# Verify everything is deployed
python3 -c "
import sys; sys.path.insert(0,'src')
from topologies.star import StarTopology
from topologies.dynamic_reputation import DynamicReputationTopology
from event_extraction.graph_builder import extract_graph_rows
from tail_fitting.powerlaw_fit import fit_all, OBS_SPECS
from tables.generate_tables import generate_all_tables
print('All imports OK')
"

# Create scratch output directories
mkdir -p /home/kavanav/mas-powerlaws/mas-results/{MARBLE,GAIA,REALM,SWE-bench}
mkdir -p /home/kavanav/mas-powerlaws/mas-results/combined
mkdir -p ~/mas-powerlaws/logs
mkdir -p ~/mas-powerlaws/slurm

================================================================================
STEP 1 — SUBMIT 4 SWEEP JOBS SIMULTANEOUSLY
================================================================================

# ── GPU 0: MARBLE ─────────────────────────────────────────────────────────────
cat > ~/mas-powerlaws/slurm/sweep_MARBLE.slurm << 'SLURM'
#!/bin/bash
#SBATCH --job-name=mas-MARBLE
#SBATCH --output=/home/kavanav/mas-powerlaws/logs/MARBLE_%j.out
#SBATCH --error=/home/kavanav/mas-powerlaws/logs/MARBLE_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal_q

cd ~/mas-powerlaws
source venv/bin/activate

python scripts/main.py --mode full \
    --benchmark MARBLE \
    --data-root /home/kavanav/mas-powerlaws/mas-results/MARBLE \
    --scales 8 16 32 64 128 256 512 \
    --seeds 0 1 2 3 4 \
    --max-tasks 5 \
    --max-concurrent 4 \
    --model gpt-4o-mini \
    --stages sweep
SLURM

# ── GPU 1: GAIA ──────────────────────────────────────────────────────────────
cat > ~/mas-powerlaws/slurm/sweep_GAIA.slurm << 'SLURM'
#!/bin/bash
#SBATCH --job-name=mas-GAIA
#SBATCH --output=/home/kavanav/mas-powerlaws/logs/GAIA_%j.out
#SBATCH --error=/home/kavanav/mas-powerlaws/logs/GAIA_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal_q

cd ~/mas-powerlaws
source venv/bin/activate

python scripts/main.py --mode full \
    --benchmark GAIA \
    --data-root /home/kavanav/mas-powerlaws/mas-results/GAIA \
    --scales 8 16 32 64 128 256 512 \
    --seeds 0 1 2 3 4 \
    --max-tasks 5 \
    --max-concurrent 4 \
    --model gpt-4o-mini \
    --stages sweep
SLURM

# ── GPU 2: REALM ─────────────────────────────────────────────────────────────
cat > ~/mas-powerlaws/slurm/sweep_REALM.slurm << 'SLURM'
#!/bin/bash
#SBATCH --job-name=mas-REALM
#SBATCH --output=/home/kavanav/mas-powerlaws/logs/REALM_%j.out
#SBATCH --error=/home/kavanav/mas-powerlaws/logs/REALM_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal_q

cd ~/mas-powerlaws
source venv/bin/activate

python scripts/main.py --mode full \
    --benchmark REALM \
    --data-root /home/kavanav/mas-powerlaws/mas-results/REALM \
    --scales 8 16 32 64 128 256 512 \
    --seeds 0 1 2 3 4 \
    --max-tasks 5 \
    --max-concurrent 4 \
    --model gpt-4o-mini \
    --stages sweep
SLURM

# ── GPU 3: SWE-bench ─────────────────────────────────────────────────────────
cat > ~/mas-powerlaws/slurm/sweep_SWEbench.slurm << 'SLURM'
#!/bin/bash
#SBATCH --job-name=mas-SWEbench
#SBATCH --output=/home/kavanav/mas-powerlaws/logs/SWEbench_%j.out
#SBATCH --error=/home/kavanav/mas-powerlaws/logs/SWEbench_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal_q

cd ~/mas-powerlaws
source venv/bin/activate

python scripts/main.py --mode full \
    --benchmark SWE-bench \
    --data-root /home/kavanav/mas-powerlaws/mas-results/SWE-bench \
    --scales 8 16 32 64 128 256 512 \
    --seeds 0 1 2 3 4 \
    --max-tasks 5 \
    --max-concurrent 4 \
    --model gpt-4o-mini \
    --stages sweep
SLURM

# Submit all 4 simultaneously
sbatch ~/mas-powerlaws/slurm/sweep_MARBLE.slurm
sbatch ~/mas-powerlaws/slurm/sweep_GAIA.slurm
sbatch ~/mas-powerlaws/slurm/sweep_REALM.slurm
sbatch ~/mas-powerlaws/slurm/sweep_SWEbench.slurm

# Confirm submission
squeue -u kavanav

================================================================================
STEP 2 — MONITOR PROGRESS
================================================================================

# Watch job queue
watch -n 60 squeue -u kavanav

# Count completed runs per benchmark (run periodically)
for bench in MARBLE GAIA REALM SWE-bench; do
    n=$(find /home/kavanav/mas-powerlaws/mas-results/$bench -name "run_metadata.json" 2>/dev/null | wc -l)
    echo "$bench: $n / 525 runs complete"
done

# Check for failures in a specific benchmark log
tail -f ~/mas-powerlaws/logs/MARBLE_*.out

# Count failures
grep "FAIL" ~/mas-powerlaws/logs/MARBLE_*.out | wc -l

# Quick disk usage check
du -sh /home/kavanav/mas-powerlaws/mas-results/*

================================================================================
STEP 3 — IF A JOB FAILS OR TIMES OUT: RESUME
================================================================================

# Resume uses --resume flag which skips runs where run_metadata.json exists
# Just resubmit the failed benchmark job — it will pick up where it left off

python scripts/main.py --mode full \
    --benchmark MARBLE \
    --data-root /home/kavanav/mas-powerlaws/mas-results/MARBLE \
    --scales 8 16 32 64 128 256 512 \
    --seeds 0 1 2 3 4 \
    --max-tasks 5 \
    --max-concurrent 4 \
    --model gpt-4o-mini \
    --stages sweep \
    --resume

================================================================================
STEP 4 — POST-SWEEP: COMBINED ANALYSIS (after ALL 4 jobs finish)
================================================================================

# Verify all 4 benchmarks completed
for bench in MARBLE GAIA REALM SWE-bench; do
    n=$(find /home/kavanav/mas-powerlaws/mas-results/$bench -name "run_metadata.json" 2>/dev/null | wc -l)
    echo "$bench: $n runs  (expected 525)"
done

# Run combined analysis — extract + fit + tables + figures
python scripts/main.py --mode analyse \
    --data-roots \
        /home/kavanav/mas-powerlaws/mas-results/MARBLE \
        /home/kavanav/mas-powerlaws/mas-results/GAIA \
        /home/kavanav/mas-powerlaws/mas-results/REALM \
        /home/kavanav/mas-powerlaws/mas-results/SWE-bench \
    --out /home/kavanav/mas-powerlaws/mas-results/combined

# This runs all 5 stages:
#   extract  → claim_nodes.csv, claim_edges.csv, observables_pooled.json, run_metrics.csv
#   analyse  → fit_results.csv, fits_by_group.csv
#   tables   → table1-5 + appendix A-B  (.tex + .csv)
#   figures  → 21 figures (.pdf + .png)

================================================================================
STEP 5 — EXPECTED OUTPUTS
================================================================================

/home/kavanav/mas-powerlaws/mas-results/combined/
  extracted/
    observables_pooled.json        ← pooled H1 observables (all benchmarks)
    observables_by_group.json      ← per (topology, benchmark, N)
    claim_nodes.csv                ← 2,100 runs × ~20 claims avg = ~42K rows
    claim_edges.csv                ← ~100K propagation edges
    agent_edges.csv                ← ~50K agent influence edges
    run_graph_summary.csv          ← 2,100 rows
    run_metrics.csv                ← 2,100 rows (H2)
    scaling_table.csv              ← H3 xmax/gini vs N
  analysis/
    fit_results.csv                ← pooled PL fits
    fits_by_group.csv              ← per (topology, benchmark, N) fits  ← Tables 2+3
  tables/
    table1_pl_fits.tex/.csv        ← PL statistics (primary observables)
    table2_alpha_across_settings.tex/.csv  ← α vs topology × N
    table3_model_comparison.tex/.csv       ← PL vs alternatives
    table4_influence_concentration.tex/.csv ← Gini, top-5%, xmax per topology
    table5_quality_efficiency.tex/.csv     ← H2 scores per topology
    appendix_a_tail_interpretation.tex/.csv
    appendix_b_experiment_design.tex/.csv
  figures/
    *.pdf  *.png                   ← 21 figures

================================================================================
STEP 6 — SANITY CHECKS BEFORE ANALYSIS
================================================================================

# Spot-check event structure
find /home/kavanav/mas-powerlaws/mas-results/MARBLE -name "events.jsonl" | head -3 | xargs -I{} sh -c 'echo "=== {} ===" && head -1 {}'

# Verify H2 metadata has key fields
python3 -c "
import json, pathlib
p = next(pathlib.Path('/home/kavanav/mas-powerlaws/mas-results/MARBLE').rglob('run_metadata.json'))
m = json.loads(p.read_text())
for k in ['task_score','tokens_total','topology','num_agents']:
    print(f'{k}: {m.get(k)}')
extra = m.get('extra',{})
for k in ['claim_participation_rate','architecture']:
    print(f'extra.{k}: {extra.get(k)}')
"

# Verify graph structure is clean (check for placeholder nodes)
python3 -c "
import pandas as pd
cn = pd.read_csv('/home/kavanav/mas-powerlaws/mas-results/combined/extracted/claim_nodes.csv')
total = len(cn)
placeholders = cn['is_placeholder'].sum()
print(f'Claim nodes: {total:,}  Placeholders: {placeholders} ({100*placeholders/total:.1f}%)')
print('Topologies:', cn['topology'].unique().tolist())
print('Benchmarks:', cn['benchmark'].unique().tolist())
print('N values:', sorted(cn['num_agents'].unique().tolist()))
"

================================================================================