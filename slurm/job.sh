#!/bin/bash -l
#SBATCH --account=project_465003463
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00
set -euo pipefail
RUN_ROOT=$1
STAGE=$2
source "$RUN_ROOT/environment.sh"
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_WANDB_MODE=offline
export SINGULARITYENV_LIE_GROUP="$(basename "$RUN_ROOT")"
export SINGULARITYENV_OMP_NUM_THREADS=8
export SINGULARITYENV_XDG_CACHE_HOME="${TMPDIR:-/tmp}/lie_ssl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="$SINGULARITYENV_XDG_CACHE_HOME/miopen"
mkdir -p "$SINGULARITYENV_XDG_CACHE_HOME/miopen"
cd "$RUN_ROOT/source"
if [[ "$STAGE" == smoke ]]; then
    srun singularity exec -B /scratch/project_465003463 "$SIF" python -m unittest discover -s tests -v
    for ((SETTING_INDEX=4*SLURM_ARRAY_TASK_ID; SETTING_INDEX<4*SLURM_ARRAY_TASK_ID+4; SETTING_INDEX++)); do
        srun singularity exec -B /scratch/project_465003463 "$SIF" \
            python -u -m lie_ssl.cli smoke --root "$RUN_ROOT" --data-root "$DATA_ROOT" --index "$SETTING_INDEX"
    done
    exit 0
fi
srun singularity exec -B /scratch/project_465003463 "$SIF" \
    python -u -m lie_ssl.cli "$STAGE" --root "$RUN_ROOT" --data-root "$DATA_ROOT" --index "${SLURM_ARRAY_TASK_ID:-0}"
