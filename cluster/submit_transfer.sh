#!/bin/bash
# Submit the CAMB transfer table Delta_l(k), l = 2..2000, as SLURM jobs (camb_inputs.py transfer).
#
#   cluster/submit_transfer.sh test      # one short job: timing, memory, check vs stored table, batch plan
#   cluster/submit_transfer.sh run       # array job, one (k block x l chunk) task per index, then a merge job
#   cluster/submit_transfer.sh missing   # resubmit only tasks without an output file, then a merge job
#   cluster/submit_transfer.sh merge     # merge job only
#
# Run it on the login node from anywhere, with the Python environment that has camb active
# (sbatch passes the environment on), or set ENV_SETUP to the command that activates it.
# The settings below can be overridden on the command line, e.g.
#   TASK_TIME=08:00:00 MAX_RUNNING=50 cluster/submit_transfer.sh run
#   DRY_RUN=1 cluster/submit_transfer.sh run            # print the sbatch commands only
set -euo pipefail
cd "$(dirname "$0")/.."                                 # codes_dev

export ELLS=${ELLS:-2:2000}
export N_L_CHUNKS=${N_L_CHUNKS:-1}                      # keep 1: every l chunk re-runs CAMB for its k block
export OUT=${OUT:-data/Tgamma_lk_camb_source_lmax2000.npz}
export ENV_SETUP=${ENV_SETUP:-}                         # e.g. 'source ~/miniforge3/bin/activate camb'
PARTITION=${PARTITION:-}                                # empty: the cluster's default partition
TASK_TIME=${TASK_TIME:-06:00:00}                        # ~2x the longest task predicted by the test
TASK_MEM=${TASK_MEM:-2G}
MAX_RUNNING=${MAX_RUNNING:-100}                         # array tasks running at the same time
DRY_RUN=${DRY_RUN:-0}

[[ "$OUT" == *.npz ]] || { echo "OUT must end in .npz" >&2; exit 1; }
mkdir -p output/slurm
part=()
[[ -n "$PARTITION" ]] && part=(--partition="$PARTITION")

submit() {                                              # sbatch, printing the job id
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[dry run] MODE=$MODE sbatch $*" >&2
        echo 0
    else
        sbatch --parsable "$@" | cut -d";" -f1      # "id;cluster" on multi-cluster setups
    fi
}

task_list() {                                           # task ids to run, most expensive (highest k) first
    python - "$1" <<'EOF'
import os
import sys
from pathlib import Path
import numpy as np
import camb_inputs as ci
import config
ells, n_l_chunks, out = ci.parse_ells(os.environ["ELLS"]), int(os.environ["N_L_CHUNKS"]), Path(os.environ["OUT"])
try:
    with np.load(config.TRANSFER_FILE) as f:
        eta0 = float(f["eta0"][0])
except OSError:
    eta0 = 14150.094557673423                           # CAMB's eta0 for the config cosmology
n = len(ci.tasks(len(ci.k_grid(eta0)), ells, n_l_chunks))
task_dir = out.with_name(out.stem + "_tasks")
done = {t for t in range(n) if (task_dir / f"task{t:05d}.npz").is_file()}
todo = [t for t in range(n - 1, -1, -1) if sys.argv[1] == "run" or t not in done]
print("\n".join(map(str, todo)))
print(f"{n} tasks (TRANSFER_K_BLOCK = {config.TRANSFER_K_BLOCK}, l chunks = {n_l_chunks}); "
      f"{len(done)} already in {task_dir}; submitting {len(todo)}", file=sys.stderr)
EOF
}

submit_merge() {
    MODE=merge submit "${part[@]}" --job-name=camb-transfer-merge --time=00:30:00 --mem=4G \
        --output=output/slurm/transfer_merge_%j.out "$@" cluster/transfer.sbatch
}

case "${1:-}" in
test)
    id=$(MODE=test submit "${part[@]}" --job-name=camb-transfer-test --time=01:00:00 --mem=4G \
        --output=output/slurm/transfer_test_%j.out cluster/transfer.sbatch)
    echo "test job $id -> output/slurm/transfer_test_$id.out"
    ;;
run | missing)
    export TASK_LIST="output/slurm/transfer_tasks_$(date +%Y%m%d_%H%M%S).txt"
    task_list "$1" > "$TASK_LIST"
    n=$(wc -l < "$TASK_LIST")
    if (( n == 0 )); then
        echo "no tasks to run"
        exit 0
    fi
    max_array=$( (scontrol show config 2>/dev/null || true) | awk '/^MaxArraySize/ {print $3}')
    if [[ -n "$max_array" ]] && (( n > max_array )); then
        echo "$n tasks > MaxArraySize $max_array: raise TRANSFER_K_BLOCK in config.py" >&2
        exit 1
    fi
    array=$(MODE=task submit "${part[@]}" --job-name=camb-transfer --array="0-$((n - 1))%$MAX_RUNNING" \
        --time="$TASK_TIME" --mem="$TASK_MEM" --output=output/slurm/transfer_%A_%a.out cluster/transfer.sbatch)
    merge=$(submit_merge --dependency=afterany:"$array")
    echo "array job $array ($n tasks, list $TASK_LIST); merge job $merge runs when it ends"
    echo "watch:  squeue -u \$USER    usage:  sacct -j $array --format=JobID%20,State,Elapsed,MaxRSS"
    ;;
merge)
    merge=$(submit_merge)
    echo "merge job $merge -> output/slurm/transfer_merge_$merge.out"
    ;;
*)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
