#!/bin/bash -l
#SBATCH --job-name=nTE-disj
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=results/disjoint-%j.out
#SBATCH --error=results/disjoint-%j.err

# Disjoint replicates: each of the 100 sets is optimized against the candidate
# universe minus every row already published, so the sets share no controls and
# the effective replicate count is 100 by construction. Compared against strict
# min-W1 selection and against the diversity-aware tolerances of route D, all on
# the same target, seeds and bootstrap targets.

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6

B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
SCRATCH="${TMPDIR:-/tmp}/nTE-disj-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
rsync -a "$B/all_te_snp_age_interval_store/" "$SCRATCH/interval_store/"
echo "staged"; date

python bootstrap_target_matcher.py \
  --store "$SCRATCH/interval_store" \
  --target results/targets/in_gene_75draw \
  --seed-sets results/matches/in_gene_75draw \
  --candidate-rows results/candidate-rows-75draw.npy \
  --output results/bootstrap_matches/in_gene_75draw_disjoint \
  --work-dir results/work-disjoint --resume \
  --replicates 100 --closest-restarts 2 --diverse-restarts 1 \
  --min-epochs 10 --max-epochs 50 --patience 5 \
  --disjoint-replicates --seed 1002 2>&1 | tail -3
date
rm -rf "$SCRATCH"
