#!/bin/bash -l
#SBATCH --job-name=nTE-scale
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=results/scale-probe-%j.out
#SBATCH --error=results/scale-probe-%j.err

# Scaling probe at n=35,512 (8.7x the in-gene target). Measures reads/s, peak
# RSS, and epochs-to-converge for the matcher at a larger target, to test the
# linear projection behind the ~23 h estimate before committing build effort to
# distributed execution. Needs a target and a seed library at this size first,
# so both are built here; only step 3 is the measurement.

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6

B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
STORE=$B/all_te_snp_age_interval_store
POS=$B/run.combined.99.age.max6mis.nomispo.rna.struc.te.sorted.pos.txt
SCRATCH="${TMPDIR:-/tmp}/nTE-scale-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
echo "=== staging ==="; date
rsync -a "$STORE/" "$SCRATCH/interval_store/"

echo "=== step 1: target (35,512 sites) ==="; date
/usr/bin/time -v python te_age_target.py \
  --store "$SCRATCH/interval_store" --te-positions "$POS" \
  --output results/targets/rna_struc_75draw \
  --bootstrap-replicates 10000 --acceptance-quantile 0.50 --bin-width 1000 \
  --missing-position-policy drop --seed 2002 2>&1 | grep -vE '^\s*(Average|Voluntary|Involuntary|Signals|Socket|Swaps|Page size)'

echo "=== step 2: minimal seed library (3 sets) ==="; date
/usr/bin/time -v python sample_age_matched_controls.py \
  --store "$SCRATCH/interval_store" --target results/targets/rna_struc_75draw \
  --candidate-rows results/candidate-rows-75draw.npy \
  --output results/matches/rna_struc_75draw_seed3 \
  --work-dir results/work-scale-seed --resume \
  --sets 3 --chains 3 --sets-per-chain 1 --workers 3 --seed 2002 \
  --burnin-accepted-sweeps 1 --sample-accepted-sweeps 1 2>&1 | grep -vE '^\s*(Average|Voluntary|Involuntary|Signals|Socket|Swaps|Page size)'

echo "=== step 3: MEASUREMENT — 5 replicates x 1 restart ==="; date
/usr/bin/time -v python bootstrap_target_matcher.py \
  --store "$SCRATCH/interval_store" --target results/targets/rna_struc_75draw \
  --seed-sets results/matches/rna_struc_75draw_seed3 \
  --candidate-rows results/candidate-rows-75draw.npy \
  --output results/bootstrap_matches/rna_struc_scale_probe \
  --work-dir results/work-scale-probe --resume \
  --replicates 5 --closest-restarts 1 --diverse-restarts 0 \
  --min-epochs 10 --max-epochs 50 --patience 5 --seed 2002 2>&1 | tail -40
date
rm -rf "$SCRATCH"
