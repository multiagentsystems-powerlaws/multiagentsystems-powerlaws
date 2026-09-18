#!/bin/bash
#SBATCH --job-name=mas_analysis
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_analysis_%j.out
#SBATCH --error=logs/slurm_analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kavanav@vt.edu

module purge
source /home/kavanav/mas-powerlaws/venv/bin/activate
cd /home/kavanav/mas-powerlaws
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

RESULTS=/scratch/kavanav/mas-results
FIGURES=/scratch/kavanav/mas-results/figures

echo "Starting analysis + figure generation: $(date)"

# Merge + fit
python scripts/run_analysis.py \
    --data-roots \
        $RESULTS/MARBLE \
        $RESULTS/GAIA \
        $RESULTS/REALM \
        $RESULTS/SWE-bench \
    --out $RESULTS/combined

# Generate all paper figures
python scripts/run_visualizations.py \
    --data-roots \
        $RESULTS/MARBLE \
        $RESULTS/GAIA \
        $RESULTS/REALM \
        $RESULTS/SWE-bench \
    --out $FIGURES/main

# Generate all tables
python scripts/generate_tables.py \
    --data-roots \
        $RESULTS/MARBLE \
        $RESULTS/GAIA \
        $RESULTS/REALM \
        $RESULTS/SWE-bench \
    --out $RESULTS/tables

echo "Analysis done: $(date)"
echo "Figures: $FIGURES/main/"
echo "Tables:  $RESULTS/tables/"
