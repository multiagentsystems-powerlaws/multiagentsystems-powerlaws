#!/bin/bash
#SBATCH --job-name=mas_REALM
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_REALM_%j.out
#SBATCH --error=logs/slurm_REALM_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kavanav@vt.edu

module purge
source /home/kavanav/mas-powerlaws/venv/bin/activate
cd /home/kavanav/mas-powerlaws
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

echo "Starting REALM sweep: $(date)"
python scripts/run_sweep.py \
    --benchmark REALM \
    --data-root /scratch/kavanav/mas-results/REALM \
    --all-topologies \
    --scales 8 16 32 64 128 \
    --seeds 0 1 2 \
    --model gpt-4o-mini \
    --max-concurrent 4 \
    --resume

echo "REALM sweep done: $(date)"