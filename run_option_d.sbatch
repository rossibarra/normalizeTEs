#!/bin/bash -l
#SBATCH --job-name=nTE-optD
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=results/option-d-%j.out
#SBATCH --error=results/option-d-%j.err

# Route D: diversity-aware restart selection. Identical to the published
# production run except that a replicate may publish any restart within
# --selection-tolerance of its own best W1, taking the one whose rows are least
# reused by earlier replicates. Everything else -- seeds, bootstrap targets,
# restart computation -- is unchanged, so the comparison isolates the selection
# rule. Three tolerances, because the trade-off curve is the result, not any
# single point.

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6

B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
SCRATCH="${TMPDIR:-/tmp}/nTE-optD-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
rsync -a "$B/all_te_snp_age_interval_store/" "$SCRATCH/interval_store/"
echo "staged"; date

for TOL in 0.02 0.05 0.15; do
  TAG="${TOL/./p}"
  echo "================ selection tolerance $TOL ================"; date
  python bootstrap_target_matcher.py \
    --store "$SCRATCH/interval_store" \
    --target results/targets/in_gene_75draw \
    --seed-sets results/matches/in_gene_75draw \
    --candidate-rows results/candidate-rows-75draw.npy \
    --output results/bootstrap_matches/in_gene_75draw_divsel${TAG} \
    --work-dir results/work-divsel${TAG} --resume \
    --replicates 100 --closest-restarts 2 --diverse-restarts 1 \
    --min-epochs 10 --max-epochs 50 --patience 5 \
    --selection-tolerance "$TOL" --seed 1002 2>&1 | tail -2
done
date
rm -rf "$SCRATCH"
