#!/bin/bash
# Job body for the CAMB transfer run. Submit it with cluster/submit_transfer.sh, which sets
# MODE (test | task | merge), the resources, and ELLS, N_L_CHUNKS, OUT, TASK_LIST, ENV_SETUP.
#SBATCH --job-name=camb-transfer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail
: "${MODE:?MODE is set by cluster/submit_transfer.sh}"
cd "$SLURM_SUBMIT_DIR"                      # codes_dev
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n "${ENV_SETUP:-}" ]]; then set +u; eval "$ENV_SETUP"; set -u; fi
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+ index $SLURM_ARRAY_TASK_ID} mode $MODE"

case "$MODE" in
test)
    git log -1 --format='code: %h %s' 2>/dev/null || true
    git status --short --untracked-files=no 2>/dev/null || true
    python -u cluster/transfer_test.py --ells "$ELLS" --n-l-chunks "$N_L_CHUNKS"
    ;;
task)
    task=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")
    file="${OUT%.npz}_tasks/task$(printf %05d "$task").npz"
    if [[ -s "$file" ]]; then
        echo "$file exists; nothing to do"
        exit 0
    fi
    python -u camb_inputs.py transfer --ells "$ELLS" --n-l-chunks "$N_L_CHUNKS" --task "$task" --out "$OUT"
    ;;
merge)
    python -u camb_inputs.py transfer --ells "$ELLS" --n-l-chunks "$N_L_CHUNKS" --merge --out "$OUT"
    ;;
*)
    echo "unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
echo "$(date '+%F %T') done in ${SECONDS} s"
