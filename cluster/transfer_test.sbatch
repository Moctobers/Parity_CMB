#!/bin/bash
# Pre-flight check before cluster/transfer.sbatch (~10 min): versions, CAMB time and memory per k,
# comparison with the stored l <= 50 table, projection timing and a K_BLOCK plan.
#   sbatch cluster/transfer_test.sbatch
#SBATCH --job-name=camb-transfer-test
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/transfer_test_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamin.hou@lmu.de

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
git log -1 --format='code: %h %s' 2>/dev/null || true
python -u cluster/transfer_test.py --ells 2:2000 --n-l-chunks 1
