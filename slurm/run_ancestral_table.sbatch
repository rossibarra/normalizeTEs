#!/bin/bash -l
#SBATCH --job-name=nTE-ancestral
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=ancestral-%A_%a.out
#SBATCH --error=ancestral-%A_%a.err

# Build the ancestral-polarity table that Phi-SFS needs for control SNPs.
#
# The input VCF is not polarized: SINGER infers the ancestral allele, so it
# lives in each ARG's site table and differs between draws. This accumulates,
# for every store row, how many draws called each of A/C/G/T ancestral.
#
# Runs either as one job over all draws or as a SLURM array. Decompression
# dominates -- about one minute per draw, and reading the ancestral states is
# free once the tables are in memory -- so 75 draws is roughly 75 minutes in a
# single job, or a few minutes wide as an array.
#
#   ARRAY MODE   sbatch --array=0-14 --export=ALL,STORE=...,TREES="/path/*.tsz",\
#                OUTPUT=results/ancestral-parts,PER_TASK=5 slurm/run_ancestral_table.sbatch
#                then gather with MERGE=1 and EXPECT_DRAWS set to the total.
#
#   SINGLE JOB   sbatch --export=ALL,STORE=...,TREES="/path/*.tsz",\
#                OUTPUT=results/ancestral-75draw slurm/run_ancestral_table.sbatch
#
# Variables:
#   STORE         completed interval store                        [required]
#   TREES         glob of tree-sequence draws, quoted             [required unless MERGE]
#   OUTPUT        table directory to publish; must not exist      [required]
#   PER_TASK      draws per array task                            (default: all)
#   MERGE         set to 1 to gather PARTS into OUTPUT instead of building
#   PARTS         glob of part directories to merge               [required if MERGE]
#   EXPECT_DRAWS  total draws the merged table must contain; without it a
#                 partial gather cannot be distinguished from a complete one

set -euo pipefail
# sbatch copies the job script into /var/spool/slurmd, so BASH_SOURCE does not
# point at the checkout. SLURM_SUBMIT_DIR does; the BASH_SOURCE form is the
# fallback for running this script directly.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${STORE:?set STORE to the completed interval-store directory}"
: "${OUTPUT:?set OUTPUT to the table directory to publish}"
cd "$PROJECT"
# Batch jobs get a non-login shell, where `module` does not exist.
. "$PROJECT/slurm/slurm_conda_bootstrap.sh"

if [[ "${MERGE:-0}" == "1" ]]; then
    : "${PARTS:?set PARTS to a glob of part directories}"
    # shellcheck disable=SC2086
    python -m normalize_tes.build_ancestral_states --store "$STORE" --output "$OUTPUT" \
        --merge $PARTS ${EXPECT_DRAWS:+--expect-draws "$EXPECT_DRAWS"}
    exit 0
fi

: "${TREES:?set TREES to a quoted glob of .tsz draws}"
# Without nullglob an unmatched pattern survives as its own literal text, so
# the array is non-empty and this check passes on a glob that matched nothing.
shopt -s nullglob
# shellcheck disable=SC2206
tree_files=( $TREES )
shopt -u nullglob
[[ ${#tree_files[@]} -gt 0 ]] || { echo "TREES matched no files: $TREES" >&2; exit 1; }

slice=()
out="$OUTPUT"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${PER_TASK:-}" ]]; then
    start=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
    stop=$(( start + PER_TASK ))
    slice=(--draws "${start}:${stop}")
    out="${OUTPUT}/part-$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
    mkdir -p "$OUTPUT"
fi

echo "store   $STORE"
echo "draws   ${#tree_files[@]} matched${slice:+, slice ${slice[1]}}"
echo "output  $out"
date
python -m normalize_tes.build_ancestral_states --store "$STORE" --output "$out" \
    "${slice[@]}" "${tree_files[@]}"
date
