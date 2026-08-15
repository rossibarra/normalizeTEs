#!/usr/bin/env bash
#SBATCH --job-name=age-match
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/age-match-%A_%a.out
#SBATCH --error=logs/age-match-%A_%a.err

set -euo pipefail

# Required submission-time variables:
#   PROJECT=/quobyte/.../normalizeTE
#   STORE=/quobyte/.../snp_interval_store
#   MANIFEST=/quobyte/.../te_manifest.tsv
#   AGE_MATCH_TASK_COUNT=10  (must equal the SLURM array size)
: "${PROJECT:?set PROJECT to the repository path on Quobyte}"
: "${STORE:?set STORE to the completed interval-store directory on Quobyte}"
: "${MANIFEST:?set MANIFEST to the TE manifest on Quobyte}"
: "${AGE_MATCH_TASK_COUNT:?set AGE_MATCH_TASK_COUNT to the array size}"
: "${TMPDIR:?Farm must provide per-job TMPDIR under /local/scratch}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is unavailable}"

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

# Each array task stages once, then processes its round-robin manifest shard.
# This avoids random interval reads from Quobyte during the swap chains.
rsync -a --delete "$STORE/" "$TMPDIR/interval_store/"

cd "$PROJECT"
python run_age_match_manifest.py sample \
    --manifest "$MANIFEST" \
    --store "$TMPDIR/interval_store" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --task-count "$AGE_MATCH_TASK_COUNT" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --resume
