#!/bin/bash -l
#SBATCH --job-name=nTE-individual-ages
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=individual-ages-%A_%a.out
#SBATCH --error=individual-ages-%A_%a.err

# Combined posterior age distribution of the derived alleles each individual
# carries, weighted by the posterior probability that the carried allele is
# the derived one.
#
# One pass over the VCF dominates the runtime, and per-sample cost is linear in
# the number of samples, so restrict SAMPLES when the VCF is sample-rich. Run
# one task per chromosome VCF and gather the parts with MERGE=1: the parts are
# summed, which is exact, because the mixture is a sum over sites.
#
#   ARRAY MODE   sbatch --array=0-9 --export=ALL,STORE=...,POLARITY=...,\
#                VCFS="/path/chr*.vcf.gz",OUTPUT=results/individual-ages-parts \
#                slurm/run_individual_age_spectrum.sbatch
#                then gather with MERGE=1.
#
#   SINGLE JOB   sbatch --export=ALL,STORE=...,POLARITY=...,VCFS="/path/all.vcf.gz",\
#                OUTPUT=results/individual-ages slurm/run_individual_age_spectrum.sbatch
#
# Variables:
#   STORE         completed interval store                        [required]
#   POLARITY      per-draw polarity table from build_draw_polarity
#                                                                 [required unless MERGE]
#   VCFS          glob of biallelic VCFs, quoted                  [required unless MERGE]
#   OUTPUT        result directory to publish; must not exist     [required]
#   SAMPLES_FILE  file of sample names, one per line              (default: every sample)
#   INCLUDE       'chrom position' file restricting the sites analysed
#   EXCLUDE       'chrom position' file of sites to drop, e.g. all TE positions
#   MIN_DRAWS     minimum draws that both date and orient a site  (default: 8)
#   WEIGHTING     site | dosage                                   (default: site)
#   BIN_SCALE     steps | log | linear                            (default: steps)
#   BIN_STEPS     quoted WIDTH:LIMIT segments in generations, used by the
#                 default 'steps' scale
#                 (default: 100:10000 1000:100000 5000:500000 10000:2000000
#                  100000:10000000)
#   BIN_MIN       first inner bin edge, in generations            (log/linear only)
#   BIN_MAX       last inner bin edge, in generations             (log/linear only)
#   N_BINS        inner bins between BIN_MIN and BIN_MAX          (log/linear only)
#   MERGE         set to 1 to sum PARTS into OUTPUT instead of scanning
#   PARTS         glob of part directories to merge               [required if MERGE]

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${STORE:?set STORE to the completed interval-store directory}"
: "${OUTPUT:?set OUTPUT to the result directory to publish}"
cd "$PROJECT"
. "$PROJECT/slurm/slurm_conda_bootstrap.sh"

if [[ "${MERGE:-0}" == "1" ]]; then
    : "${PARTS:?set PARTS to a glob of part directories}"
    # shellcheck disable=SC2086
    python -m normalize_tes.individual_age_spectrum --store "$STORE" \
        --output "$OUTPUT" --merge $PARTS
    exit 0
fi

: "${POLARITY:?set POLARITY to the per-draw polarity table}"
: "${VCFS:?set VCFS to a quoted glob of VCF files}"
shopt -s nullglob
# shellcheck disable=SC2206
vcf_files=( $VCFS )
shopt -u nullglob
[[ ${#vcf_files[@]} -gt 0 ]] || { echo "VCFS matched no files: $VCFS" >&2; exit 1; }

out="$OUTPUT"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    index="$SLURM_ARRAY_TASK_ID"
    [[ "$index" -lt "${#vcf_files[@]}" ]] || {
        echo "array index $index exceeds the ${#vcf_files[@]} matched VCFs" >&2; exit 1; }
    vcf_files=( "${vcf_files[$index]}" )
    out="${OUTPUT}/part-$(printf '%03d' "$index")"
    mkdir -p "$OUTPUT"
fi

echo "store     $STORE"
echo "polarity  $POLARITY"
echo "vcfs      ${#vcf_files[@]}"
echo "output    $out"
date
# shellcheck disable=SC2086  # BIN_STEPS must split into one word per segment
python -m normalize_tes.individual_age_spectrum \
    --store "$STORE" --draw-polarity "$POLARITY" --output "$out" \
    --vcf "${vcf_files[@]}" \
    ${SAMPLES_FILE:+--samples-file "$SAMPLES_FILE"} \
    ${INCLUDE:+--include-positions "$INCLUDE"} \
    ${EXCLUDE:+--exclude-positions "$EXCLUDE"} \
    --min-usable-draws "${MIN_DRAWS:-8}" \
    --allele-weighting "${WEIGHTING:-site}" \
    --bin-scale "${BIN_SCALE:-steps}" \
    ${BIN_STEPS:+--bin-steps $BIN_STEPS} \
    ${BIN_MIN:+--bin-min "$BIN_MIN"} \
    ${BIN_MAX:+--bin-max "$BIN_MAX"} \
    ${N_BINS:+--n-bins "$N_BINS"}
date
