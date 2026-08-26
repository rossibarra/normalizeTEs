#!/usr/bin/env bash
#SBATCH --job-name=nTE-polarity-mask
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=polarity-mask-%j.out
#SBATCH --error=polarity-mask-%j.err

# Builds the TE polarity mask consumed by normalize_tes.te_age_target.
#
# For every TE site in TARGET, records which posterior draws called the
# insertion allele derived, as biology says it almost always is. A draw that
# called it ancestral placed the mutation on a different branch of the ARG and
# recorded that branch's age, so its age estimates a different event.
#
# The tree list is taken from the store's own metadata["inputs"], so mask
# columns are indexed by the store's draw_id and every draw is covered.
# normalize_tes.te_age_target refuses a mask that does not cover the whole store, because
# an uncovered draw is indistinguishable there from a flipped one.
#
# Required:
#   STORE     interval store the target rows index into
#   TARGET    target directory supplying te_row_indices.npy
#   OUTPUT    destination directory for the mask
# Optional:
#   ABSENCE_ALLELE   allele encoding TE absence (default A)
#
# Measured: 4,067 sites x 75 draws took 1 h 21 m; cost is dominated by
# decompressing each tsz once, not by the number of sites.

set -euo pipefail
# sbatch copies the job script into /var/spool/slurmd, so BASH_SOURCE does not
# point at the checkout. SLURM_SUBMIT_DIR does; the BASH_SOURCE form is the
# fallback for running this script directly.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${STORE:?set STORE to the interval-store directory}"
: "${TARGET:?set TARGET to the target directory}"
: "${OUTPUT:?set OUTPUT to the mask directory to publish}"
ABSENCE_ALLELE="${ABSENCE_ALLELE:-A}"

cd "$PROJECT"
# Batch jobs get a non-login shell, where `module` does not exist.
. "$PROJECT/slurm/slurm_conda_bootstrap.sh"

threads="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"

# The store names its own source trees in draw_id order; using that list rather
# than a glob is what guarantees full coverage.
mapfile -t TREES < <(python -c "
import json, sys
m = json.load(open(sys.argv[1] + '/metadata.json'))
for e in sorted(m['inputs'], key=lambda x: x['draw_id']):
    print(e['path'])
" "$STORE")
echo "store lists ${#TREES[@]} draws"; date

python -m normalize_tes.build_te_polarity_mask \
    --store "$STORE" \
    --target "$TARGET" \
    --output "$OUTPUT" \
    --absence-allele "$ABSENCE_ALLELE" \
    "${TREES[@]}"
date
