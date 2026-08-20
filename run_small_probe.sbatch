#!/bin/bash -l
#SBATCH --job-name=nTE-small
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/small-probe-%j.out
#SBATCH --error=results/small-probe-%j.err

# Size dependence of the matching floor, holding TE composition fixed. The
# 4,067 and 35,466 measurements used different TE sets, so size and composition
# were confounded. These are random subsamples of the same in-gene set, so only
# n varies. SWAP_SAMPLER_HPC_HOWTO.md §10 anticipates a ~500-SNP target, which
# is the regime this probes.

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6

B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
SCRATCH="${TMPDIR:-/tmp}/nTE-small-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
rsync -a "$B/all_te_snp_age_interval_store/" "$SCRATCH/interval_store/"
echo "staged"; date

for N in 600 1500; do
  echo "================ n=$N ================"; date
  python te_age_target.py --store "$SCRATCH/interval_store" \
    --te-positions results/ingene_subsample_${N}.pos.txt \
    --output results/targets/ingene_sub${N} \
    --bootstrap-replicates 10000 --acceptance-quantile 0.50 --bin-width 1000 \
    --missing-position-policy drop --seed 3002

  python sample_age_matched_controls.py --store "$SCRATCH/interval_store" \
    --target results/targets/ingene_sub${N} \
    --candidate-rows results/candidate-rows-75draw.npy \
    --output results/matches/ingene_sub${N}_seed3 \
    --work-dir results/work-sub${N}-seed --resume \
    --sets 3 --chains 3 --sets-per-chain 1 --workers 3 --seed 3002 \
    --burnin-accepted-sweeps 1 --sample-accepted-sweeps 1 2>&1 | tail -3

  python bootstrap_target_matcher.py --store "$SCRATCH/interval_store" \
    --target results/targets/ingene_sub${N} \
    --seed-sets results/matches/ingene_sub${N}_seed3 \
    --candidate-rows results/candidate-rows-75draw.npy \
    --output results/bootstrap_matches/ingene_sub${N}_probe \
    --work-dir results/work-sub${N}-probe --resume \
    --replicates 20 --closest-restarts 1 --diverse-restarts 0 \
    --min-epochs 10 --max-epochs 50 --patience 5 --seed 3002 2>&1 | tail -3
done
date
rm -rf "$SCRATCH"
