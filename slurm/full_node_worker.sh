#!/bin/bash
set -euo pipefail
RUN_ROOT=${1:?Run root required}
STAGE=${2:?Stage required}
RANK_INDEX=${SLURM_PROCID:?Slurm rank required}
GROUP_INDEX=${SLURM_ARRAY_TASK_ID:-0}
[[ "$RANK_INDEX" =~ ^[0-7]$ ]] || { echo "Invalid worker rank" >&2; exit 2; }
case "$STAGE" in
    train|evaluate) SETTING_INDEX=$RANK_INDEX ;;
    probe)
        [[ "$GROUP_INDEX" =~ ^[0-9]+$ ]] && (( GROUP_INDEX < 20 )) || { echo "Invalid probe group" >&2; exit 2; }
        SETTING_INDEX=$((8*GROUP_INDEX+RANK_INDEX)) ;;
    *) echo "Invalid stage" >&2; exit 2 ;;
esac
printf 'stage=%s group=%s rank=%s index=%s\n' "$STAGE" "$GROUP_INDEX" "$RANK_INDEX" "$SETTING_INDEX"
if [[ "${LIE_DRY_RUN:-0}" == 1 ]]; then exit 0; fi
source "$RUN_ROOT/environment.sh"
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_WANDB_MODE=offline
export SINGULARITYENV_LIE_GROUP="$(basename "$RUN_ROOT")"
export SINGULARITYENV_OMP_NUM_THREADS=7
export SINGULARITYENV_MKL_NUM_THREADS=7
export SINGULARITYENV_XDG_CACHE_HOME="${TMPDIR:-/tmp}/lie_ssl_${SLURM_JOB_ID}_${RANK_INDEX}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="$SINGULARITYENV_XDG_CACHE_HOME/miopen"
mkdir -p "$SINGULARITYENV_MIOPEN_USER_DB_PATH"
cd "$RUN_ROOT/source"
exec singularity exec -B /scratch/project_465003463 "$SIF" \
    python -u - "$STAGE" --root "$RUN_ROOT" --data-root "$DATA_ROOT" --index "$SETTING_INDEX" <<'PY'
import os
import torch
if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Each full-node worker must see exactly one ROCm GPU')
print('GPU binding:', {k:os.environ.get(k) for k in ('SLURM_PROCID','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')}, flush=True)
from lie_ssl.cli import main
main()
PY
