#!/bin/bash -l
#SBATCH --job-name=nTE-loggrid
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=results/loggrid-%j.out
#SBATCH --error=results/loggrid-%j.err

# Search-grid control for the log-age arm. Identical to
# results/bootstrap_matches/in_gene_75draw_disjoint in every respect -- linear
# W1, same linear target, same seed, so the same bootstrap replicates -- except
# that the coarse swap screen is the geometric sub-sample of the exact grid used
# by the log-age arm. Any young-end improvement visible here is attributable to
# the finer young-end screen rather than to the metric.

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
SCRATCH="${TMPDIR:-/tmp}/nTE-loggrid-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
rsync -a "$B/all_te_snp_age_interval_store/" "$SCRATCH/interval_store/"
echo "staged store"; date

python bootstrap_target_matcher.py \
  --store "$SCRATCH/interval_store" \
  --target results/targets/in_gene_75draw \
  --seed-sets results/matches/in_gene_75draw \
  --candidate-rows results/candidate-rows-75draw.npy \
  --output results/bootstrap_matches/in_gene_75draw_loggrid_disjoint \
  --work-dir results/work-loggrid-disjoint --resume \
  --replicates 100 --closest-restarts 2 --diverse-restarts 1 \
  --min-epochs 10 --max-epochs 50 --patience 5 \
  --disjoint-replicates \
  --distance linear --search-grid-spacing log \
  --seed 1002 2>&1 | tail -5
date
rm -rf "$SCRATCH"
