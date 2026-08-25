#!/usr/bin/env bash
#SBATCH --job-name=nTE-bootstrap-match
#SBATCH --account=jrigrp
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=bootstrap-match-%j.out
#SBATCH --error=bootstrap-match-%j.err

# Scheduler output goes to the submission directory, not logs/. SLURM opens
# these files before any line of this script runs, so a directory the script
# creates later cannot exist yet: on a fresh clone the job would fail before
# reaching mkdir. Move the logs afterwards if you want them filed away.

# The canonical production launcher for bootstrap-target control matching.
#
# Submit with sbatch, never srun: an srun started from an interactive shell dies
# with that shell, which is how the first production attempt was lost at 48
# minutes. Completed replicate bundles under WORK_DIR survive job death, so the
# identical resubmission continues from where it stopped (--resume is always
# passed below).
#
# Environment variables, all overridable at submit time:
#
#   PROJECT          repository checkout (default: the directory holding this file)
#   STORE            canonical interval store on durable storage        [required]
#   TARGET           target directory built by te_age_target.py         [required]
#   OUTPUT           bundle directory to publish; must not exist        [required]
#   CANDIDATE_ROWS   .npy from build_candidate_rows.py, or "all" to use
#                    every eligible non-target row       (default: required path)
#   WORK_DIR         durable resume directory       (default: OUTPUT.work)
#   REPLICATES       control sets to publish                    (default: 100)
#   RESTARTS         stratified restarts per replicate            (default: 3)
#   SEED             global seed                                (default: 1002)
#   TE_POSITIONS     build TARGET from this position file if TARGET is absent
#   ACCEPTANCE_QUANTILE        bootstrap quantile for the target threshold (0.50)
#   MISSING_POSITION_POLICY    error|drop for target construction     (error)
#   TE_POLARITY_MASK           polarity mask dir, when building a target (unset)
#   MAX_FLIPPED_FRACTION       discard TEs flipped above this fraction  (unset)
#
# Example:
#
#   sbatch --export=ALL,STORE=/quobyte/project/data/snp_interval_store,\
#   TARGET=results/targets/in_gene,OUTPUT=results/bootstrap_matches/in_gene,\
#   CANDIDATE_ROWS=results/candidate-rows.npy,WORK_DIR=results/work-in-gene,SEED=1002 \
#     run_bootstrap_matching.sbatch
#
# Measured on the 75-draw production store (18.2 GB) at this allocation, for a
# 4,067-site target with 100 replicates x 3 restarts: 2 h 12 m wall clock and
# 37.7 GiB peak RSS (SLURM job 37821602). Target construction is the other
# memory peak: 16.0 GB for a 35,512-site target. Scale --time roughly linearly
# in target size and replicate count, and keep --mem at 96G unless a measured
# figure says otherwise.

set -euo pipefail

# sbatch copies the job script into /var/spool/slurmd, so BASH_SOURCE does not
# point at the checkout. SLURM_SUBMIT_DIR does; the BASH_SOURCE form is the
# fallback for running this script directly.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
: "${STORE:?set STORE to the completed interval-store directory}"
: "${TARGET:?set TARGET to the target directory}"
: "${OUTPUT:?set OUTPUT to the bundle directory to publish}"
: "${CANDIDATE_ROWS:?set CANDIDATE_ROWS to a .npy of candidate rows, or to \"all\"}"
: "${TMPDIR:?Farm must provide per-job TMPDIR under /local/scratch}"
WORK_DIR="${WORK_DIR:-${OUTPUT}.work}"
REPLICATES="${REPLICATES:-100}"
RESTARTS="${RESTARTS:-3}"
SEED="${SEED:-1002}"
TE_POSITIONS="${TE_POSITIONS:-}"
ACCEPTANCE_QUANTILE="${ACCEPTANCE_QUANTILE:-0.50}"
MISSING_POSITION_POLICY="${MISSING_POSITION_POLICY:-error}"
TE_POLARITY_MASK="${TE_POLARITY_MASK:-}"
MAX_FLIPPED_FRACTION="${MAX_FLIPPED_FRACTION:-}"

cd "$PROJECT"
mkdir -p "$TMPDIR/interval_store"

# Batch jobs get a non-login shell, where `module` does not exist.
. "$PROJECT/slurm_conda_bootstrap.sh"

threads="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$threads"
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"

# Staging is mandatory, not an optimization. Against the store on Quobyte the
# run is I/O-bound and projects to 4-21 h of pure store reads for one target;
# staged to node-local scratch the same run measured 97% CPU with 21 major page
# faults. Every job that touches the store stages it first.
required_kib=$(du -sk "$STORE" | awk '{print $1}')
free_kib=$(df -Pk "$TMPDIR" | awk 'NR==2 {print $4}')
if (( free_kib < required_kib * 6 / 5 )); then
    echo "insufficient node-local scratch: need the store size plus 20%" >&2
    exit 2
fi
echo "staging $STORE to $TMPDIR/interval_store"; date
rsync -a --delete "$STORE/" "$TMPDIR/interval_store/"
date

# A draw that mis-polarized a TE placed its mutation on a different branch, so
# the age it recorded belongs to a different event. With a mask, each TE's age
# CDF is built from its agreeing draws only.
polarity_option=()
if [[ -n "$TE_POLARITY_MASK" ]]; then
    polarity_option+=(--te-polarity-mask "$TE_POLARITY_MASK")
    if [[ -n "$MAX_FLIPPED_FRACTION" ]]; then
        polarity_option+=(--max-flipped-fraction "$MAX_FLIPPED_FRACTION")
    fi
elif [[ -n "$MAX_FLIPPED_FRACTION" ]]; then
    echo "MAX_FLIPPED_FRACTION requires TE_POLARITY_MASK" >&2
    exit 5
fi

if [[ ! -d "$TARGET" ]]; then
    if [[ -z "$TE_POSITIONS" ]]; then
        echo "TARGET $TARGET does not exist and TE_POSITIONS is unset" >&2
        exit 4
    fi
    echo "=== building target $TARGET ==="; date
    python te_age_target.py \
        --store "$TMPDIR/interval_store" \
        --te-positions "$TE_POSITIONS" \
        --output "$TARGET" \
        --scratch-dir "$TMPDIR" \
        --bootstrap-replicates 10000 \
        --acceptance-quantile "$ACCEPTANCE_QUANTILE" \
        --bin-width 1000 \
        --missing-position-policy "$MISSING_POSITION_POLICY" \
        "${polarity_option[@]}" \
        --seed "$SEED"
    date
fi

if [[ "$CANDIDATE_ROWS" == "all" ]]; then
    candidate_option=(--all-eligible)
else
    candidate_option=(--candidate-rows "$CANDIDATE_ROWS")
fi

echo "=== matching: $REPLICATES replicates x $RESTARTS restarts ==="; date
python bootstrap_target_matcher.py \
    --store "$TMPDIR/interval_store" \
    --target "$TARGET" \
    "${candidate_option[@]}" \
    --output "$OUTPUT" \
    --work-dir "$WORK_DIR" \
    --resume \
    --replicates "$REPLICATES" \
    --restarts "$RESTARTS" \
    --disjoint-replicates \
    --seed "$SEED"
date
