#!/bin/bash
# Factorisable Fisher with the full transfer function on several radial grids, to test convergence in the grid.
#
#   bash cluster/fisher_factorisable_radial_grid.sh submit     # per grid: kernels job -> Fisher array over mu nodes -> merge
#   bash cluster/fisher_factorisable_radial_grid.sh summary    # F(L) of every grid, relative to the base grid
#   DRY_RUN=1 bash cluster/fisher_factorisable_radial_grid.sh submit
#
# Run it on the login node with the Python environment active, after making the n_s = 1 C_l with
# `python camb_inputs.py spectra --lmax 2000` (inside intjob). Kernels go to
# data/kernels_lmax<LMAX>_<grid>.npz (kept and reused), results to output/fisher_transfer_lmax<LMAX>_<grid>.npz.
# Settings can be overridden on the command line, e.g.  LMAX=50 N_CHUNKS=10 bash cluster/fisher_factorisable_radial_grid.sh submit
#
# Rerun one failed Fisher task (here task 7 of grid base), then the merge once it has finished:
#   sbatch --constraint=x86-64-v3 --job-name=radial-fisher-base --array=7 --time=04:00:00 --mem=2G \
#       --output=output/slurm/fisher_radial_base_%A_%a.out cluster/run.sbatch \
#       fisher --source transfer --lmax 100 --ell-step 25 --kernels data/kernels_lmax100_base.npz \
#       --out output/fisher_transfer_lmax100_base.npz --n-chunks 20
#   sbatch --mem=2G --time=00:20:00 --output=output/slurm/fisher_radial_merge_base_%j.out cluster/run.sbatch \
#       fisher --source transfer --lmax 100 --ell-step 25 --kernels data/kernels_lmax100_base.npz \
#       --out output/fisher_transfer_lmax100_base.npz --n-chunks 20 --merge
set -euo pipefail
cd "$(dirname "$0")/.."                                 # codes_dev

LMAX=${LMAX:-100}
ELL_STEP=${ELL_STEP:-25}                                # cutoffs 25, 50, 75, 100
N_CHUNKS=${N_CHUNKS:-20}                                # Fisher array tasks per grid (mu-node chunks)
TASK_TIME=${TASK_TIME:-04:00:00}
PARTITION=${PARTITION:-}
CONSTRAINT=${CONSTRAINT:-x86-64-v3}                     # keeps jobs off the old, slow nodes
DRY_RUN=${DRY_RUN:-0}

# name and "start:stop:step" segments; base = config.RADIAL_GRID
NAMES=(base shell2p5 xmax30k outer25)
SPECS=("0:chi-500:50,chi-500:chi:5,chi:eta0:5,eta0:16000:50"
       "0:chi-500:50,chi-500:chi:2.5,chi:eta0:2.5,eta0:16000:50"
       "0:chi-500:50,chi-500:chi:5,chi:eta0:5,eta0:30000:50"
       "0:chi-500:25,chi-500:chi:5,chi:eta0:5,eta0:16000:25")

mkdir -p output/slurm
opts=()
[[ -n "$PARTITION" ]] && opts+=(--partition="$PARTITION")
[[ -n "$CONSTRAINT" ]] && opts+=(--constraint="$CONSTRAINT")

submit() {                                              # sbatch, printing the job id
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[dry run] sbatch $*" >&2
        echo 0
    else
        sbatch --parsable "$@" | cut -d";" -f1
    fi
}

case "${1:-}" in
submit)
    for i in "${!NAMES[@]}"; do
        name=${NAMES[$i]} spec=${SPECS[$i]}
        kern=data/kernels_lmax${LMAX}_${name}.npz
        out=output/fisher_transfer_lmax${LMAX}_${name}.npz
        dep=()
        if [[ -f "$kern" ]]; then
            echo "$name: reusing $kern"
        else
            kid=$(submit "${opts[@]}" --job-name="radial-kern-$name" --time=03:00:00 --mem=4G \
                --output="output/slurm/fisher_radial_kernels_${name}_%j.out" \
                cluster/run.sbatch kernels --lmax "$LMAX" --grid "$spec" --out "$kern")
            dep=(--dependency=afterok:"$kid")
        fi
        fisher=(fisher --source transfer --lmax "$LMAX" --ell-step "$ELL_STEP" --kernels "$kern" --out "$out"
                --n-chunks "$N_CHUNKS")
        fid=$(submit "${opts[@]}" "${dep[@]}" --job-name="radial-fisher-$name" --array="0-$((N_CHUNKS - 1))" \
            --time="$TASK_TIME" --mem=2G --output="output/slurm/fisher_radial_${name}_%A_%a.out" \
            cluster/run.sbatch "${fisher[@]}")
        mid=$(submit "${opts[@]}" --dependency=afterany:"$fid" --job-name="radial-merge-$name" --time=00:20:00 \
            --mem=2G --output="output/slurm/fisher_radial_merge_${name}_%j.out" cluster/run.sbatch "${fisher[@]}" --merge)
        echo "$name: kernels ${kid:-reused}, Fisher array $fid ($N_CHUNKS tasks), merge $mid"
        unset kid
    done
    echo "watch: squeue -u \$USER;  then: bash cluster/fisher_factorisable_radial_grid.sh summary"
    echo "if a Fisher task fails: see the rerun example in the header of $0"
    ;;
summary)
    python - "$LMAX" "${NAMES[@]}" <<'EOF'
import sys
from pathlib import Path
import numpy as np
lmax, names = sys.argv[1], sys.argv[2:]
results = {}
for name in names:
    path = Path(f"output/fisher_transfer_lmax{lmax}_{name}.npz")
    if not path.exists():
        print(f"{name:9s} missing ({path})")
        continue
    with np.load(path) as f:
        results[name] = (f["L"], f["F"], f["snr"])
    with np.load(f"data/kernels_lmax{lmax}_{name}.npz") as k:
        print(f"{name:9s} {len(k['x']):5d} radial nodes, x_max = {k['x'][-1]:.0f} Mpc")
if "base" not in results:
    sys.exit("no base result yet")
L, F0, snr0 = results["base"]
print("\nL      " + "".join(f"{n:>22s}" for n in results))
for i, ell in enumerate(L):
    row = f"{int(ell):<6d} " + f"{'SNR ' + format(snr0[i], '.5g'):>22s}"
    row += "".join(f"{'dF/F ' + format(F[i] / F0[i] - 1, '+.2e'):>22s}" for n, (_, F, _) in results.items() if n != "base")
    print(row)
EOF
    ;;
*)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
