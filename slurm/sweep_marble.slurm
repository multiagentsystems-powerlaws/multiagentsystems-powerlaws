#!/bin/bash
#SBATCH --job-name=mas_MARBLE
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_MARBLE_%j.out
#SBATCH --error=logs/slurm_MARBLE_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kavanav@vt.edu

# ── Environment ──────────────────────────────────────────────────
module purge
source /home/kavanav/mas-powerlaws/venv/bin/activate
cd /home/kavanav/mas-powerlaws
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# ── API keys (set these in your ~/.bashrc or here) ───────────────
# export OPENAI_API_KEY=sk-...

# ── Rate limit strategy ──────────────────────────────────
# 4 Slurm jobs × 4 concurrent runs each = 16 simultaneous API calls.
# OpenAI Tier 1: 500 RPM — should be fine.
# OpenAI Tier 2: 5000 RPM — can raise --max-concurrent to 8 per job.
# If you hit 429s, reduce --max-concurrent or stagger job start times.
# Failed runs are logged to data/failed_runs.jsonl; re-run with --resume.

# ── Run ──────────────────────────────────────────────────────────
echo "Starting MARBLE sweep: $(date)"
python scripts/run_sweep.py \
    --benchmark MARBLE \
    --data-root /scratch/kavanav/mas-results/MARBLE \
    --all-topologies \
    --scales 8 16 32 64 128 \
    --seeds 0 1 2 \
    --model gpt-4o-mini \
    --max-concurrent 4 \
    --resume

echo "MARBLE sweep done: $(date)"