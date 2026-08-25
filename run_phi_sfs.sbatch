#!/bin/bash -l
#SBATCH --job-name=nTE-phisfs
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=phi-sfs-%j.out
#SBATCH --error=phi-sfs-%j.err

# Compare the unfolded site frequency spectrum of a TE target with that of every
# matched control set.
#
# Polarity comes from two sources and neither is the VCF. TE sites are polarized
# by biology -- a TE insertion is the derived state -- and control SNPs by the
# ARG-derived table from run_ancestral_table.sbatch, applied as a posterior
# weighted mixture rather than a majority call. Running this against an
# unpolarized VCF as though REF were ancestral would mis-polarize about a third
# of all sites, which is why --ancestral-table is required.
#
# The VCF must cover every requested site: the TE target plus every control SNP
# across all sets. Those span the genome -- for the in-gene target, 94% of the
# 406,700 control sites lie outside chromosome 10 -- so a single-chromosome VCF
# will abort with a list of missing coordinates. Concatenate the per-chromosome
# inputs first if they are not already one file.
#
# Example:
#
#   sbatch --export=ALL,TARGET=results/targets/in_gene,\
#   MATCHES=results/bootstrap_matches/in_gene,VCF=/path/all.chr.vcf.gz,\
#   ANCESTRAL=results/ancestral-75draw,OUTPUT=results/phi_sfs/in_gene \
#     run_phi_sfs.sbatch
#
# Variables:
#   TARGET      target directory built by te_age_target.py        [required]
#   MATCHES     matched-control bundle                            [required]
#   VCF         genome-wide polarized-agnostic VCF                [required]
#   ANCESTRAL   table directory from run_ancestral_table.sbatch   [required]
#   OUTPUT      results directory to publish; must not exist      [required]
#   HETEROZYGOUS  error|missing for diploid calls                 (error)
#
# The scan is the dominant cost and is polarity-independent: it reads the VCF
# once, streaming, and holds only the requested sites. Memory scales with the
# number of requested sites, not with VCF size.

set -euo pipefail
# sbatch copies the job script into /var/spool/slurmd, so BASH_SOURCE does not
# point at the checkout. SLURM_SUBMIT_DIR does; the BASH_SOURCE form is the
# fallback for running this script directly.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
: "${TARGET:?set TARGET to the target directory}"
: "${MATCHES:?set MATCHES to the matched-control bundle}"
: "${VCF:?set VCF to a VCF covering every requested site}"
: "${ANCESTRAL:?set ANCESTRAL to the ancestral-state table directory}"
: "${OUTPUT:?set OUTPUT to the results directory to publish}"
HETEROZYGOUS="${HETEROZYGOUS:-error}"
cd "$PROJECT"
# Batch jobs get a non-login shell, where `module` does not exist.
. "$PROJECT/slurm_conda_bootstrap.sh"

for required in "$TARGET/metadata.json" "$MATCHES/metadata.json" \
                "$ANCESTRAL/metadata.json" "$VCF"; do
    [[ -e "$required" ]] || { echo "missing input: $required" >&2; exit 1; }
done

echo "target     $TARGET"
echo "matches    $MATCHES"
echo "ancestral  $ANCESTRAL"
echo "vcf        $VCF"
date
python phi_sfs.py \
    --target "$TARGET" \
    --matches "$MATCHES" \
    --vcf "$VCF" \
    --ancestral-table "$ANCESTRAL" \
    --output "$OUTPUT" \
    --heterozygous "$HETEROZYGOUS"
date
