#!/bin/bash -l
#SBATCH --account=project_465003463
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=mi250:8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00
set -euo pipefail
RUN_ROOT=${1:?Run root required}
STAGE=${2:?train, evaluate, or probe required}
LAUNCH_DIR=${3:-$RUN_ROOT/source/slurm}
LOG_DIR=${4:-$RUN_ROOT/logs}
case "$STAGE" in train|evaluate|probe) ;; *) echo "Invalid stage: $STAGE" >&2; exit 2;; esac
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
mkdir -p "$LOG_DIR"
cd "$RUN_ROOT/source"
# Eight independent experiments, each bound to one MI250X GCD.
# per_task:1 binding is implied by --gpus-per-task; do not use 'closest'.
srun --ntasks=8 --gpus-per-task=mi250:1 --cpus-per-task=7 --cpu-bind=cores --kill-on-bad-exit=0 \
    --output="$LOG_DIR/standard-${SLURM_JOB_ID}-${STAGE}-%t.out" \
    --error="$LOG_DIR/standard-${SLURM_JOB_ID}-${STAGE}-%t.err" \
    bash "$LAUNCH_DIR/full_node_worker.sh" "$RUN_ROOT" "$STAGE"
