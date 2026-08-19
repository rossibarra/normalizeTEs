#!/bin/bash -l
#SBATCH --job-name=nTE-deep
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=results/nTE-deep-bootstrap-%j.out
#SBATCH --error=results/nTE-deep-bootstrap-%j.err

# Detached from any interactive shell: an srun launched from a terminal dies
# when that terminal goes away, which killed the first T5 attempt at 48 minutes.
# Completed replicate bundles in --work-dir survive, so --resume continues.

set -euo pipefail
module load conda
conda activate normalizeTE

STORE=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined/all_te_snp_age_interval_store
SCRATCH="${TMPDIR:-/tmp}/nTE-t5-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
echo "staging store to $SCRATCH"; date
rsync -a "$STORE/" "$SCRATCH/interval_store/"
df -h "$SCRATCH" | tail -1

echo "--- T5: 100 replicates x 3 restarts ---"; date
python bootstrap_target_matcher.py \
  --store "$SCRATCH/interval_store" \
  --target results/targets/in_gene_75draw \
  --seed-sets results/matches/in_gene_75draw \
  --candidate-rows results/candidate-rows-75draw.npy \
  --output results/bootstrap_matches/in_gene_75draw_deepsearch \
  --work-dir results/work-deep --resume \
  --replicates 20 --closest-restarts 1 --diverse-restarts 0 \
  --min-epochs 15 --max-epochs 300 --patience 25 \
  --seed 1002
date
rm -rf "$SCRATCH"
