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

readonly source_dir=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
readonly output_dir="${source_dir}/age_store"
readonly scratch_dir="${TMPDIR:-/tmp}"

module load conda
conda activate normalizeTE

python build_snp_age_store.py \
  "${source_dir}"/run.combined.*.tsz \
  --numpy-store "${output_dir}" \
  --chrom-offsets "${source_dir}/chrom_offsets.txt" \
  --bin-width 1000 \
  --block-snps 100000 \
  --min-usable-fraction 0.1 \
  --scratch-dir "${scratch_dir}"
