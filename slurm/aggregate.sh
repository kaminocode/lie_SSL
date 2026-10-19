#!/bin/bash -l
#SBATCH --account=project_465003463
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
RUN_ROOT=$1
source "$RUN_ROOT/environment.sh"
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_OMP_NUM_THREADS=4
cd "$RUN_ROOT/source"
EXTRA=()
if [[ "${2:-}" == smoke ]]; then EXTRA+=(--smoke); fi
srun singularity exec -B /scratch/project_465003463 "$SIF" \
    python -u -m lie_ssl.cli aggregate --root "$RUN_ROOT" --data-root "$DATA_ROOT" "${EXTRA[@]}"
