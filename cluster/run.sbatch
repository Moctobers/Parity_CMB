#!/bin/bash
# Run one run.py command as a SLURM job:
#   sbatch [sbatch options] cluster/run.sbatch <run.py arguments>
# In an array job, --task $SLURM_ARRAY_TASK_ID is appended to the arguments.
# Set ENV_SETUP to the command that activates the Python environment if sbatch does not pass it on.
#SBATCH --job-name=run-py
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamin.hou@lmu.de

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                      # codes_dev
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n "${ENV_SETUP:-}" ]]; then set +u; eval "$ENV_SETUP"; set -u; fi

args=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then args+=(--task "$SLURM_ARRAY_TASK_ID"); fi
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID}: python run.py ${args[*]}"
python -u run.py "${args[@]}"
echo "$(date '+%F %T') done in ${SECONDS} s"
