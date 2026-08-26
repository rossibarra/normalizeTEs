#!/bin/bash -l
#SBATCH --job-name=normalizeTE-store
#SBATCH --account=jrigrp
#SBATCH --partition=low
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --output=results/normalizeTE-store-%j.out
#SBATCH --error=results/normalizeTE-store-%j.err

set -euo pipefail

# sbatch copies the job script into /var/spool/slurmd, so BASH_SOURCE does not
# point at the checkout. SLURM_SUBMIT_DIR does; the BASH_SOURCE form is the
# fallback for running this script directly.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

readonly source_dir=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
readonly output_dir="${source_dir}/age_store"
readonly scratch_dir="${TMPDIR:-/tmp}"

# `python -m normalize_tes.*` resolves the package from the working directory,
# so the job must run from the checkout rather than wherever it was submitted.
cd "$PROJECT"
# Batch jobs get a non-login shell, where `module` does not exist.
. "$PROJECT/slurm/slurm_conda_bootstrap.sh"

python -m normalize_tes.build_snp_age_store \
  "${source_dir}"/run.combined.*.tsz \
  --numpy-store "${output_dir}" \
  --chrom-offsets "${source_dir}/chrom_offsets.txt" \
  --bin-width 1000 \
  --block-snps 100000 \
  --min-usable-fraction 0.1 \
  --scratch-dir "${scratch_dir}"
