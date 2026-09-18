#!/bin/bash
#SBATCH --job-name=mas_SWE
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_SWE_%j.out
#SBATCH --error=logs/slurm_SWE_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kavanav@vt.edu

module purge
source /home/kavanav/mas-powerlaws/venv/bin/activate
cd /home/kavanav/mas-powerlaws
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

echo "Starting SWE-bench sweep: $(date)"
python scripts/run_sweep.py \
    --benchmark SWE-bench \
    --data-root /scratch/kavanav/mas-results/SWE-bench \
    --all-topologies \
    --scales 8 16 32 64 128 \
    --seeds 0 1 2 \
    --model gpt-4o-mini \
    --max-concurrent 4 \
    --resume

echo "SWE-bench sweep done: $(date)"