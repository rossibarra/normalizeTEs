#!/bin/bash -l
#SBATCH --job-name=nTE-logage
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=results/logage-%j.out
#SBATCH --error=results/logage-%j.err

# Log-age matching arm. Builds a bootstrap tolerance for the same 4,067-site
# in-gene TE set under the log-age CDF distance (weights d log(age + 1000)
# instead of d age), then repeats the disjoint 100-replicate match under that
# metric. The coarse swap screen switches to a geometric sub-sample of the
# exact grid, because a uniform 20,000-generation screen puts the entire lower
# quartile of the age distribution in its first cell and would be blind to the
# region the log-age metric is meant to price.
#
# Compare against results/bootstrap_matches/in_gene_75draw_disjoint (linear W1,
# linear screen) and results/bootstrap_matches/in_gene_75draw_loggrid_disjoint
# (linear W1, log screen -- isolates the screen change from the metric change).

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
SCRATCH="${TMPDIR:-/tmp}/nTE-logage-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/interval_store"
rsync -a "$B/all_te_snp_age_interval_store/" "$SCRATCH/interval_store/"
echo "staged store"; date

if [[ ! -d results/targets/in_gene_75draw_logage ]]; then
  python te_age_target.py \
    --store "$SCRATCH/interval_store" \
    --te-positions run.combined.99.age.max6mis.nomispo.rna.struc.te.ingene.pos.txt \
    --output results/targets/in_gene_75draw_logage \
    --bootstrap-replicates 10000 \
    --acceptance-quantile 0.50 \
    --bin-width 1000 \
    --missing-position-policy drop \
    --distance log-age \
    --log-age-offset 1000 \
    --scratch-dir "$SCRATCH" \
    --seed 1002
fi
echo "target built"; date

python bootstrap_target_matcher.py \
  --store "$SCRATCH/interval_store" \
  --target results/targets/in_gene_75draw_logage \
  --seed-sets results/matches/in_gene_75draw \
  --candidate-rows results/candidate-rows-75draw.npy \
  --output results/bootstrap_matches/in_gene_75draw_logage_disjoint \
  --work-dir results/work-logage-disjoint --resume \
  --replicates 100 --closest-restarts 2 --diverse-restarts 1 \
  --min-epochs 10 --max-epochs 50 --patience 5 \
  --disjoint-replicates \
  --distance log-age --log-age-offset 1000 --search-grid-spacing log \
  --seed 1002 2>&1 | tail -5
date
rm -rf "$SCRATCH"
