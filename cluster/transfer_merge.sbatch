#!/bin/bash
# Merge the 509 task files of cluster/transfer.sbatch into data/Tgamma_lk_camb_source_lmax2000.npz:
#   sbatch --dependency=afterany:<array jobid> cluster/transfer_merge.sbatch     # or plainly, once all tasks are done
# Fails with "found N task files, expected 509" if tasks are missing; list them with
#   python -c "from pathlib import Path; print(','.join(str(t) for t in range(509) if not Path(f'data/Tgamma_lk_camb_source_lmax2000_tasks/task{t:05d}.npz').is_file()))"
# and rerun them with sbatch --array=<that list> cluster/transfer.sbatch.
#SBATCH --job-name=camb-transfer-merge
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=output/slurm/transfer_merge_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiamin.hou@lmu.de

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python -u camb_inputs.py transfer --ells 2:2000 --n-l-chunks 1 --merge --out data/Tgamma_lk_camb_source_lmax2000.npz
echo "$(date '+%F %T') done in ${SECONDS} s"
