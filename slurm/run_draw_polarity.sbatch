#!/bin/bash -l
#SBATCH --job-name=nTE-draw-polarity
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=draw-polarity-%A_%a.out
#SBATCH --error=draw-polarity-%A_%a.err

# Build the per-draw ancestral-state table used by individual_age_spectrum.
#
# build_ancestral_states records how MANY draws called each base ancestral.
# This records WHICH draw called what, so an age interval can be paired with
# the polarity of the same draw. Phi-SFS does not need that; a per-individual
# derived-allele age distribution does.
#
# Cost matches the ancestral table: decompression dominates at roughly one
# minute per draw, so 75 draws is about 75 minutes in a single job, or a few
# minutes wide as an array. A part stores only its own draw columns, so an
# array does not multiply the full table by the number of tasks.
#
#   ARRAY MODE   sbatch --array=0-14 --export=ALL,STORE=...,TREES="/path/*.tsz",\
#                OUTPUT=results/draw-polarity-parts,PER_TASK=5 \
#                slurm/run_draw_polarity.sbatch
#                then gather with MERGE=1 and EXPECT_DRAWS set to the total.
#
#   SINGLE JOB   sbatch --export=ALL,STORE=...,TREES="/path/*.tsz",\
#                OUTPUT=results/draw-polarity-75draw slurm/run_draw_polarity.sbatch
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
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${STORE:?set STORE to the completed interval-store directory}"
: "${OUTPUT:?set OUTPUT to the table directory to publish}"
cd "$PROJECT"
. "$PROJECT/slurm/slurm_conda_bootstrap.sh"

if [[ "${MERGE:-0}" == "1" ]]; then
    : "${PARTS:?set PARTS to a glob of part directories}"
    # shellcheck disable=SC2086
    python -m normalize_tes.build_draw_polarity --store "$STORE" --output "$OUTPUT" \
        --merge $PARTS ${EXPECT_DRAWS:+--expect-draws "$EXPECT_DRAWS"}
    exit 0
fi

: "${TREES:?set TREES to a quoted glob of .tsz draws}"
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
python -m normalize_tes.build_draw_polarity --store "$STORE" --output "$out" \
    "${slice[@]}" "${tree_files[@]}"
date
