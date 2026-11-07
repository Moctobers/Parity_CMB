"""Pre-flight check for the CAMB transfer run on the cluster (one core, ~15-25 min).

    python cluster/transfer_test.py [--ells 2:2000] [--n-l-chunks 1] [--max-hours 4] [--max-mem-gb 4]

Run it on a compute node (cluster/submit_transfer.sh test, or inside intjob), not on the
login node. It prints
  1. the node, its CPU and the package versions;
  2. CAMB's eta0 and the (k, eta) grids;
  3. CAMB T_source time and memory per k, for 32 k spread over the grid and the 32 highest k;
  4. Delta_l(k), l = 2..50, at those k against the stored table (config.TRANSFER_FILE);
  5. the projection time per k over the production l chunks, at 11 points of the k grid
     (timed on every 20th eta node and scaled up; the cost per node does not depend on the source);
  6. task time, memory and core-hours for a range of config.TRANSFER_K_BLOCK, with a recommendation;
  7. one cheap production task (task 0, l = 2..3) through camb_inputs.run_task.
"""
import argparse
import math
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camb_inputs as ci  # noqa: E402
import config  # noqa: E402

N_CAMB_TEST = 32          # k values per CAMB timing call
N_PROJ_POINTS = 11        # points of the k grid where the projection is timed
ETA_EVERY = 20            # projection timed on every 20th eta node
K_BLOCKS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024)


def peak_mb():
    """Peak resident memory of this process so far (Linux reports kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def max_array_size():
    try:
        text = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True).stdout
        return int(re.search(r"MaxArraySize\s*=\s*(\d+)", text).group(1))
    except (OSError, AttributeError, ValueError):
        return 1001


def hours(seconds):
    return f"{seconds / 3600:6.2f} h"


def section(title):
    print(f"\n== {title}", flush=True)


def camb_source(results, k, eta):
    """The CAMB call of camb_inputs.run_task, timed."""
    t = time.perf_counter()
    source = np.asarray(results.get_time_evolution(
        k, eta, vars=["T_source"], lAccuracyBoost=config.TRANSFER_L_ACCURACY_BOOST))[:, :, 0]
    return source, time.perf_counter() - t


def load_stored():
    """The stored transfer table (config.TRANSFER_FILE), or None."""
    path = Path(config.TRANSFER_FILE)
    if not path.exists():
        return None
    with np.load(path) as f:
        return dict(T=f["Tgamma_lk"], k=f["k_arr"], ells=f["ells_sparse"].astype(int), eta0=float(f["eta0"][0]),
                    camb=str(f["camb_version"][0]) if "camb_version" in f.files else "unknown")


def compare_with_stored(stored, picks, sources, k_test, eta, eta0):
    """Delta_l(k) from this CAMB at the stored k values, against the stored rows l = 2..50."""
    rows = np.array([l for l in range(2, 51) if l in stored["ells"]])
    old_rows = stored["T"][np.searchsorted(stored["ells"], rows)]
    scale = np.abs(old_rows).max(axis=1)
    worst = 0.0
    for name, idx in picks:
        new = ci.project(sources[name], k_test[idx], eta, eta0, rows)
        err = np.abs(new - old_rows[:, idx]).max(axis=1) / scale
        worst = max(worst, err.max())
        print(f"{name:>20}: max_k |new - stored| / max_k |stored|: worst {err.max():.1e} "
              f"(l = {rows[err.argmax()]}), median over l {np.median(err):.1e}")
    if worst > 1e-3:
        print("WARNING: differences above 1e-3; check the CAMB version and camb_params before the full run")
    elif worst > 1e-6:
        print("note: small differences (1e-6..1e-3), e.g. from the CAMB version or set_for_lmax(2000) vs 1000")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ells", type=ci.parse_ells, default=ci.parse_ells("2:2000"), help="production l range")
    parser.add_argument("--n-l-chunks", type=int, default=1, help="production l chunks")
    parser.add_argument("--max-hours", type=float, default=4.0, help="longest task you want (predicted, 1 core)")
    parser.add_argument("--max-mem-gb", type=float, default=4.0, help="largest memory per task you want")
    args = parser.parse_args()
    ells, chunks = args.ells, np.array_split(args.ells, args.n_l_chunks)

    section("1. node and versions")
    t = time.perf_counter()
    import camb
    import scipy
    t_import = time.perf_counter() - t
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"host {socket.gethostname()}, CPU {cpu_model()}, {cores} cores visible, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, SLURM job {os.environ.get('SLURM_JOB_ID', '-')}")
    print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, camb {camb.__version__}")
    if "SLURM_JOB_ID" not in os.environ:
        print("WARNING: not inside a SLURM job; the login node is not meant for this")

    section("2. CAMB background and grids")
    t = time.perf_counter()
    results = camb.get_background(ci.camb_params(max(2000, int(ells[-1]))))
    t_background = time.perf_counter() - t
    eta0 = float(results.conformal_time(0))
    k, eta = ci.k_grid(eta0), ci.eta_grid(eta0)
    print(f"get_background {t_background:.1f} s; eta0 = {eta0:.8f} Mpc; {len(k)} k "
          f"(k eta0 = {k[0] * eta0:g}..{k[-1] * eta0:g}); {len(eta)} eta nodes")
    stored = load_stored()
    if stored is None:
        print(f"{config.TRANSFER_FILE} not found: copy it here for the check in section 4")
    else:
        print(f"stored table: camb {stored['camb']}, eta0 = {stored['eta0']:.8f} "
              f"(relative difference {abs(eta0 - stored['eta0']) / stored['eta0']:.1e}), {len(stored['k'])} k")
    # The k grid scales with eta0, so the CAMB test runs at the stored k values when the grids match in length.
    same_grid = stored is not None and len(stored["k"]) == len(k)
    k_test = stored["k"] if same_grid else k
    if same_grid:
        print(f"CAMB test below uses the stored k values (max |k/k_stored - 1| = {np.abs(k / k_test - 1).max():.1e})")

    section("3. CAMB T_source")
    picks = [("spread over k", np.unique(np.linspace(0, len(k) - 1, N_CAMB_TEST).round().astype(int))),
             ("highest k", np.arange(len(k) - N_CAMB_TEST, len(k)))]
    sources, per_k = {}, []
    base = peak_mb()
    for i, (name, idx) in enumerate(picks):
        sources[name], dt = camb_source(results, k_test[idx], eta)
        per_k.append(dt / len(idx))
        print(f"{len(idx)} k, {name:>13}: {dt:6.1f} s ({dt / len(idx):.2f} s per k)", flush=True)
        if i == 0:
            mb_per_k = (peak_mb() - base) / len(idx)
    camb_per_k = max(per_k)
    n_vars = len(camb.model.evolve_names)
    print(f"memory: {mb_per_k:.1f} MB per k on top of {base:.0f} MB (CAMB fills all {n_vars} evolve "
          f"variables, {n_vars * len(eta) * 8 / 2**20:.1f} MB per k, before returning T_source)")

    section("4. Delta_l(k), l = 2..50, against the stored table")
    if stored is None:
        print("skipped: no stored table")
    elif not same_grid:
        print(f"skipped: the stored table has {len(stored['k'])} k, this grid {len(k)}")
    else:
        compare_with_stored(stored, picks, sources, k_test, eta, eta0)
    del sources

    section(f"5. projection time over l = {ells[0]}..{ells[-1]} in {len(chunks)} chunk(s)")
    starts = np.unique(np.linspace(0, len(k) - 4, N_PROJ_POINTS).round().astype(int))
    eta_sub = eta[::ETA_EVERY]
    cost = np.zeros((len(chunks), len(starts)))          # core-s per k on the full eta grid
    rng = np.random.default_rng(0)
    for j, s in enumerate(starts):
        kk = k[s:s + 4]
        source = rng.standard_normal((len(kk), len(eta_sub)))
        for c, chunk in enumerate(chunks):
            t = time.perf_counter()
            ci.project(source, kk, eta_sub, eta0, chunk)
            cost[c, j] = (time.perf_counter() - t) / len(kk) * len(eta) / len(eta_sub)
        print(f"k eta0 = {kk[0] * eta0:8.1f}: {cost[:, j].sum():8.1f} s per k", flush=True)
    proj = np.array([np.interp(np.arange(len(k)), starts + 1.5, row) for row in cost])   # (chunk, k)
    print(f"projection, all k and l: {proj.sum() / 3600:.0f} core-h")

    section("6. batch plan: one core per task, tasks = k blocks x l chunks")
    startup = t_import + t_background + 5.0
    limit = max_array_size()
    print(f"per task: start-up {startup:.0f} s + CAMB {camb_per_k:.2f} s per k + projection; "
          f"MaxArraySize {limit}")
    print(" K_BLOCK   tasks   longest   median   core-h   memory")
    plan = []
    for kb in K_BLOCKS:
        t_task = np.array([startup + camb_per_k * len(proj[c, s:s + kb]) + proj[c, s:s + kb].sum()
                           for s in range(0, len(k), kb) for c in range(len(chunks))])
        mem_gb = (base + mb_per_k * kb + 100) / 1024
        plan.append((kb, len(t_task), t_task.max(), np.median(t_task), t_task.sum() / 3600, mem_gb))
    fits = [p for p in plan if p[2] <= args.max_hours * 3600 and p[5] <= args.max_mem_gb and p[1] <= limit]
    best = max(fits, key=lambda p: p[0]) if fits else min(plan, key=lambda p: p[2])
    for p in plan:
        mark = (" <- recommended" if p is best else "") + (" (config now)" if p[0] == config.TRANSFER_K_BLOCK else "")
        print(f"{p[0]:8d} {p[1]:7d} {hours(p[2])} {hours(p[3])} {p[4]:8.0f} {p[5]:6.1f} GB{mark}")
    task_time = math.ceil(2 * best[2] / 1800) * 1800                 # 2x the longest, in half hours
    task_mem = max(1, math.ceil(1.5 * best[5]))                       # 1.5x, whole GB
    print(f"\nrecommended: TRANSFER_K_BLOCK = {best[0]} in config.py (now {config.TRANSFER_K_BLOCK}), then\n"
          f"  TASK_TIME={task_time // 3600:02d}:{task_time % 3600 // 60:02d}:00 TASK_MEM={task_mem}G "
          f"N_L_CHUNKS={args.n_l_chunks} cluster/submit_transfer.sh run")
    if not fits:
        print(f"WARNING: no K_BLOCK meets --max-hours {args.max_hours} and --max-mem-gb {args.max_mem_gb}; "
              "the line above is the shortest option")

    section("7. one production task through camb_inputs.run_task")
    if config.TRANSFER_K_BLOCK > 128:
        print(f"skipped: TRANSFER_K_BLOCK = {config.TRANSFER_K_BLOCK} needs "
              f"~{(base + mb_per_k * config.TRANSFER_K_BLOCK) / 1024:.1f} GB; set it as recommended first")
    else:
        out_dir = config.OUTPUT / "transfer_test" / "tasks"
        t = time.perf_counter()
        ci.run_task(results, 0, np.arange(2, 4), 1, out_dir, camb_version=camb.__version__)
        with np.load(out_dir / "task00000.npz") as f:
            good = f["T"].shape == (2, min(config.TRANSFER_K_BLOCK, len(k))) and np.isfinite(f["T"]).all()
        print(f"{time.perf_counter() - t:.0f} s; output {'OK' if good else 'BAD'}")

    print(f"\nfinished in {time.perf_counter() - T0:.0f} s; peak memory {peak_mb():.0f} MB")


T0 = time.perf_counter()
if __name__ == "__main__":
    main()
