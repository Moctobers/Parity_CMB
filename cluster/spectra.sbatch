#!/bin/bash
# Lensed C_l^TT (l <= 2000) and chi_* from CAMB, with the config cosmology (n_s = 1):
#   sbatch cluster/spectra.sbatch          (from codes_dev, with the camb environment active)
# Writes data/Cl_TT.npy and data/chi_star.npy. Command-line sbatch options override the lines below.
#SBATCH --job-name=camb-spectra
#SBATCH --partition=small
#SBATCH --constraint=x86-64-v3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm/spectra_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamin.hou@lmu.de

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
if [[ -n "${ENV_SETUP:-}" ]]; then set +u; eval "$ENV_SETUP"; set -u; fi
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID}, ${OMP_NUM_THREADS} threads"
python -u camb_inputs.py spectra --lmax 2000
echo "$(date '+%F %T') done in ${SECONDS} s"
