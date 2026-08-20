#!/bin/bash -l
#SBATCH --job-name=nTE-sweep
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=results/size-sweep-%j.out
#SBATCH --error=results/size-sweep-%j.err

# Gate 4 and gate 7 at the sizes most categories actually are. The 96/100 QC
# pass rate and the diversity deficit were both measured at 4,067 sites, near
# the top of the real range. This runs the full production configuration
# (100 replicates x 3 restarts) with the threshold-scaled QC cap at three
# smaller sizes, so both gates can be judged where the programme lives.

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6

B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
SCRATCH="${TMPDIR:-/tmp}/nTE-sweep-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
rsync -a "$B/all_te_snp_age_interval_store/" "$SCRATCH/interval_store/"
echo "staged"; date

for N in 600 1500; do
  echo "================ n=$N : full seed library ================"; date
  python sample_age_matched_controls.py --store "$SCRATCH/interval_store" \
    --target results/targets/ingene_sub${N} \
    --candidate-rows results/candidate-rows-75draw.npy \
    --output results/matches/ingene_sub${N}_full \
    --work-dir results/work-sweep-${N}-seed --resume \
    --sets 100 --chains 10 --sets-per-chain 10 --workers 5 --seed 4002 \
    --burnin-accepted-sweeps 1 --sample-accepted-sweeps 1 2>&1 | tail -2

  echo "================ n=$N : production matcher ================"; date
  python bootstrap_target_matcher.py --store "$SCRATCH/interval_store" \
    --target results/targets/ingene_sub${N} \
    --seed-sets results/matches/ingene_sub${N}_full \
    --candidate-rows results/candidate-rows-75draw.npy \
    --output results/bootstrap_matches/ingene_sub${N}_prod \
    --work-dir results/work-sweep-${N}-prod --resume \
    --replicates 100 --closest-restarts 2 --diverse-restarts 1 \
    --min-epochs 10 --max-epochs 50 --patience 5 --seed 4002 2>&1 | tail -2
done
date
rm -rf "$SCRATCH"
