#!/bin/bash
# CAMB transfer table Delta_l(k), l = 2..2000, as an array job: task i = k block i of
# config.TRANSFER_K_BLOCK = 32 values (509 blocks), all l. A task whose output file exists is skipped.
#   sbatch cluster/transfer.sbatch                           # all 509 tasks, at most 100 at a time
#   sbatch --array=3,17,250 cluster/transfer.sbatch          # only these tasks (e.g. after a timeout)
#   sbatch --dependency=afterany:<jobid> cluster/transfer_merge.sbatch
# Run from codes_dev with the camb environment active. If TRANSFER_K_BLOCK changes, the number of
# tasks changes too: python -c "import camb_inputs as c, config; print(len(range(0, 16266, config.TRANSFER_K_BLOCK)))"
#SBATCH --job-name=camb-transfer
#SBATCH --partition=small
#SBATCH --constraint=x86-64-v3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --array=0-508%100
#SBATCH --output=output/slurm/transfer_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamin.hou@lmu.de

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=data/Tgamma_lk_camb_source_lmax2000.npz
task=$SLURM_ARRAY_TASK_ID
file="${OUT%.npz}_tasks/task$(printf %05d "$task").npz"
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID} task $task"
if [[ -s "$file" ]]; then echo "$file exists; nothing to do"; exit 0; fi
python -u camb_inputs.py transfer --ells 2:2000 --n-l-chunks 1 --task "$task" --out "$OUT"
echo "$(date '+%F %T') done in ${SECONDS} s"
