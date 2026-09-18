#!/bin/bash
#SBATCH --job-name=mas_single_agent
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_single_agent_%j.out
#SBATCH --error=logs/slurm_single_agent_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kavanav@vt.edu

module purge
source /home/kavanav/mas-powerlaws/venv/bin/activate
cd /home/kavanav/mas-powerlaws
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

echo "Starting single-agent baseline: $(date)"
python scripts/run_single_agent_baseline.py \
    --benchmarks MARBLE GAIA REALM SWE-bench \
    --data-root /scratch/kavanav/mas-results/single_agent \
    --model gpt-4o-mini \
    --seeds 0 1 2 \
    --max-tasks 20 \
    --resume

echo "Single-agent baseline done: $(date)"
