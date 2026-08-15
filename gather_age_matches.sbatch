#!/usr/bin/env bash
#SBATCH --job-name=age-match-gather
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/age-match-gather-%A_%a.out
#SBATCH --error=logs/age-match-gather-%A_%a.err

set -euo pipefail

: "${PROJECT:?set PROJECT to the repository path on Quobyte}"
: "${STORE:?set STORE to the completed interval-store directory on Quobyte}"
: "${MANIFEST:?set MANIFEST to the TE manifest on Quobyte}"
: "${AGE_MATCH_GATHER_TASK_COUNT:?set to the number of manifest targets}"
: "${TMPDIR:?Farm must provide per-job TMPDIR under /local/scratch}"
AGE_MATCH_CHAINS="${AGE_MATCH_CHAINS:-10}"
AGE_MATCH_SETS_PER_CHAIN="${AGE_MATCH_SETS_PER_CHAIN:-10}"

[[ "${SLURM_ARRAY_TASK_COUNT:?}" == "$AGE_MATCH_GATHER_TASK_COUNT" ]] || {
    echo "array size $SLURM_ARRAY_TASK_COUNT != gather task count $AGE_MATCH_GATHER_TASK_COUNT" >&2
    exit 3
}

mkdir -p "$PROJECT/logs" "$TMPDIR/interval_store"
if type module >/dev/null 2>&1; then
    module load conda
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "conda is unavailable" >&2
    exit 90
fi
conda activate normalizeTE
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

required_kib=$(du -sk "$STORE" | awk '{print $1}')
free_kib=$(df -Pk "$TMPDIR" | awk 'NR==2 {print $4}')
if (( free_kib < required_kib * 6 / 5 )); then
    echo "insufficient node-local scratch: need store size plus 20%" >&2
    exit 2
fi
rsync -a --delete "$STORE/" "$TMPDIR/interval_store/"

cd "$PROJECT"
python run_age_match_manifest.py gather \
    --manifest "$MANIFEST" \
    --store "$TMPDIR/interval_store" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --task-count "$AGE_MATCH_GATHER_TASK_COUNT" \
    --scratch-dir "$TMPDIR" \
    --chains "$AGE_MATCH_CHAINS" \
    --sets-per-chain "$AGE_MATCH_SETS_PER_CHAIN"
