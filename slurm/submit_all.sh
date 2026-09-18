#!/bin/bash
# Submit all 4 benchmark sweeps as independent Slurm jobs.
# Each runs on its own GPU/node with a separate data-root.
# Run from ~/mas-powerlaws/

set -e
mkdir -p logs

echo "Submitting 4 benchmark sweeps..."

JID_MARBLE=$(sbatch --parsable slurm/sweep_marble.slurm)
echo "  MARBLE:    job $JID_MARBLE"

JID_GAIA=$(sbatch --parsable slurm/sweep_gaia.slurm)
echo "  GAIA:      job $JID_GAIA"

JID_REALM=$(sbatch --parsable slurm/sweep_realm.slurm)
echo "  REALM:     job $JID_REALM"

JID_SWE=$(sbatch --parsable slurm/sweep_swebench.slurm)
echo "  SWE-bench: job $JID_SWE"

# Submit analysis job that waits for ALL 4 to finish
sbatch --dependency=afterok:${JID_MARBLE}:${JID_GAIA}:${JID_REALM}:${JID_SWE} \
       slurm/run_analysis.slurm
echo "  Analysis:  submitted (waits for all 4)"

echo ""
echo "Monitor: squeue -u kavanav"
echo "Logs:    tail -f logs/slurm_MARBLE_*.out"
