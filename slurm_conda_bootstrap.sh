# Activate the project conda environment inside a SLURM batch job.
#
# Source this; do not execute it.
#
# sbatch runs the job script in a NON-LOGIN shell. `module` is a shell function
# defined by /etc/profile.d/modules.sh, which only a login shell sources, so in
# a batch job `module` does not exist and conda is not on PATH. A launcher that
# calls `module load conda` directly therefore dies on the first line it runs,
# two seconds in, with a bare "command not found" -- or, behind a
# `type module` guard, with a misleading "conda is unavailable".
#
# The module init scripts are not written against `set -u`, so the guard is
# lifted while they are sourced and restored afterwards.
#
# Exits 90 if no conda can be found, which callers document as the
# environment-bootstrap failure code.

__nte_conda_bootstrap() {
    local restore_u=0
    case "$-" in *u*) restore_u=1; set +u ;; esac

    if ! type module >/dev/null 2>&1; then
        local init
        for init in /etc/profile.d/modules.sh "${MODULESHOME:-}/init/bash"; do
            if [ -r "$init" ]; then . "$init"; break; fi
        done
    fi
    type module >/dev/null 2>&1 && module load conda

    if ! command -v conda >/dev/null 2>&1; then
        local base
        for base in /cvmfs/hpc.ucdavis.edu/sw/conda/root "$HOME/miniconda3" "$HOME/anaconda3"; do
            if [ -r "$base/etc/profile.d/conda.sh" ]; then . "$base/etc/profile.d/conda.sh"; break; fi
        done
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is unavailable: no module system and no conda installation found" >&2
        return 90
    fi

    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${NTE_CONDA_ENV:-normalizeTE}" || return 90

    [ "$restore_u" = 1 ] && set -u
    return 0
}
__nte_conda_bootstrap || exit 90
