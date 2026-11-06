"""CAMB inputs: lensed C_l^TT and chi_*, and the temperature transfer Delta_l(k).

    python camb_inputs.py spectra --lmax 2000
    python camb_inputs.py transfer --ells 2:4 --out data/transfer_check.npz    # all tasks, then merge
    python camb_inputs.py transfer --ells 2:2000 --n-l-chunks 8 --task 5      # one task (array index)
    python camb_inputs.py transfer --ells 2:2000 --n-l-chunks 8 --merge

Delta_l(k) = int deta S_T(k, eta) j_l[k (eta0 - eta)]  (Smith et al. 2015, App. D),
with CAMB's T_source on a uniform eta grid. Each k is independent, so the work
is cut into (k block) x (l chunk) tasks. A task asks CAMB for S_T on its k block
only, projects onto its l chunk, and writes one file; --merge assembles them.
Memory per task is set by config.TRANSFER_K_BLOCK; CPU time by the l chunk.
Task files go to <out>_tasks/ next to --out (default config.TRANSFER_FILE), and an
existing table is only replaced with --overwrite.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import spherical_jn

import config
from kernels import trapezoid_weights

COSMOLOGY = dict(H0=config.H0, ombh2=config.OMBH2, omch2=config.OMCH2,
                 tau=config.TAU, A_s=config.A_S, n_s=config.N_S)


def camb_params(lmax):
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=config.H0, ombh2=config.OMBH2, omch2=config.OMCH2, mnu=0.06,
                       omk=0.0, tau=config.TAU)
    pars.InitPower.set_params(As=config.A_S, ns=config.N_S, r=0.0, pivot_scalar=config.K_PIVOT)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1, lens_margin=150, k_eta_fac=2.5)
    pars.WantCls, pars.WantTransfer, pars.DoLensing = True, True, True
    pars.DoLateRadTruncation = False
    acc = pars.Accuracy
    acc.AccuracyBoost, acc.lAccuracyBoost, acc.BessIntBoost, acc.KmaxBoost = 2, 2, 30, 3
    acc.IntTolBoost, acc.TimeStepBoost, acc.SourcekAccuracyBoost, acc.BesselBoost = 4, 2, 5, 5
    acc.IntkAccuracyBoost, acc.TransferkBoost, acc.lSampleBoost = 5, 2, 50
    acc.AccurateBB = acc.AccurateReionization = acc.AccuratePolarization = False
    return pars


def spectra(lmax):
    """Save lensed dimensionless C_l^TT (l = LMIN..lmax) and chi_* = eta0 - tau_maxvis."""
    import camb
    results = camb.get_results(camb_params(lmax))
    eta0 = float(results.conformal_time(0))
    cls = results.get_lensed_scalar_cls(CMB_unit=None, raw_cl=True)
    np.save(config.CL_TT_FILE, cls[config.LMIN:lmax + 1, 0])
    np.save(config.CHI_STAR_FILE, np.array([eta0 - float(results.tau_maxvis)]))
    print(f"saved {config.CL_TT_FILE} and {config.CHI_STAR_FILE}; eta0 = {eta0:.6f} Mpc")


def k_grid(eta0):
    """Delta k = min(epsilon k, kappa0) from k_eta_min/eta0 to k_eta_max/eta0."""
    k = [config.TRANSFER_K_ETA_MIN / eta0]
    k_max = config.TRANSFER_K_ETA_MAX / eta0
    while k[-1] < k_max:
        k.append(min(k[-1] + min(config.TRANSFER_K_EPSILON * k[-1], config.TRANSFER_KAPPA0), k_max))
    return np.array(k)


def eta_grid(eta0):
    """Uniform eta from one step (CAMB rejects eta = 0) to eta0, which is a node."""
    step = config.TRANSFER_DELTA_ETA
    eta = step * np.arange(1, int(np.floor(eta0 / step)) + 1)
    return np.append(eta[eta < eta0], eta0)


def project(source, k, eta, eta0, ells, k_chunk=64):
    """Delta[l, k] = sum_eta w_eta S(k, eta) j_l(k (eta0 - eta)), trapezoid in eta."""
    w = trapezoid_weights(eta)
    out = np.zeros((len(ells), len(k)))
    for s in range(0, len(k), k_chunk):
        arg = np.outer(k[s:s + k_chunk], eta0 - eta)
        weighted = source[s:s + k_chunk] * w
        for i, ell in enumerate(ells):
            out[i, s:s + k_chunk] = np.sum(weighted * spherical_jn(int(ell), arg), axis=1)
    return out


def tasks(n_k, ells, n_l_chunks):
    """All (k slice, l chunk) pairs, in task order."""
    k_blocks = [slice(s, min(s + config.TRANSFER_K_BLOCK, n_k))
                for s in range(0, n_k, config.TRANSFER_K_BLOCK)]
    return [(kb, lc) for kb in k_blocks for lc in np.array_split(ells, n_l_chunks)]


def run_task(results, task, ells, n_l_chunks, out_dir, camb_version="unknown"):
    eta0 = float(results.conformal_time(0))
    k, eta = k_grid(eta0), eta_grid(eta0)
    kb, chunk = tasks(len(k), ells, n_l_chunks)[task]
    source = np.asarray(results.get_time_evolution(
        k[kb], eta, vars=["T_source"], lAccuracyBoost=config.TRANSFER_L_ACCURACY_BOOST))[:, :, 0]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"task{task:05d}.npz"
    np.savez(path, T=project(source, k[kb], eta, eta0, chunk), k_start=kb.start, k_stop=kb.stop,
             ells=chunk, settings=_settings(eta0, ells, n_l_chunks), camb_version=camb_version)
    print(f"saved {path}: k[{kb.start}:{kb.stop}], l = {chunk[0]}..{chunk[-1]}")


def _settings(eta0, ells, n_l_chunks):
    return np.array([eta0, ells[0], ells[-1], len(ells), n_l_chunks, config.TRANSFER_K_BLOCK,
                     config.TRANSFER_K_ETA_MIN, config.TRANSFER_K_ETA_MAX, config.TRANSFER_KAPPA0,
                     config.TRANSFER_K_EPSILON, config.TRANSFER_DELTA_ETA,
                     config.TRANSFER_L_ACCURACY_BOOST, *COSMOLOGY.values()])


def merge(ells, n_l_chunks, in_dir, out, overwrite=False):
    if Path(out).exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    files = sorted(Path(in_dir).glob("task*.npz"))
    first = np.load(files[0])
    eta0 = float(first["settings"][0])
    k = k_grid(eta0)
    expected = tasks(len(k), ells, n_l_chunks)
    if len(files) != len(expected):
        raise RuntimeError(f"found {len(files)} task files, expected {len(expected)}")
    T, filled = np.zeros((len(ells), len(k))), np.zeros((len(ells), len(k)), dtype=bool)
    for path in files:
        with np.load(path) as f:
            if not np.array_equal(f["settings"], first["settings"]):
                raise RuntimeError(f"{path} was made with different settings")
            rows = np.searchsorted(ells, f["ells"])
            T[rows, f["k_start"]:f["k_stop"]] = f["T"]
            filled[rows, f["k_start"]:f["k_stop"]] = True
    if not filled.all():
        raise RuntimeError("some (l, k) entries are missing")
    np.savez_compressed(
        out, Tgamma_lk=T, k_arr=k, ells_sparse=ells, eta0=[eta0], complete=[True],
        cosmology_parameter_names=list(COSMOLOGY), cosmology_parameter_values=list(COSMOLOGY.values()),
        delta_eta=[config.TRANSFER_DELTA_ETA], k_eta_min=[config.TRANSFER_K_ETA_MIN],
        k_eta_max=[config.TRANSFER_K_ETA_MAX], kappa0=[config.TRANSFER_KAPPA0],
        k_epsilon=[config.TRANSFER_K_EPSILON], l_accuracy_boost=[config.TRANSFER_L_ACCURACY_BOOST],
        camb_version=[str(first["camb_version"])])
    print(f"saved {out}: {len(ells)} rows, {len(k)} k values")


def parse_ells(text):
    lo, hi = map(int, text.split(":"))
    return np.arange(lo, hi + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("spectra", "transfer"))
    parser.add_argument("--lmax", type=int, default=2000, help="spectra: maximum multipole")
    parser.add_argument("--ells", type=parse_ells, default=parse_ells("2:50"), help="transfer rows lo:hi")
    parser.add_argument("--n-l-chunks", type=int, default=1)
    parser.add_argument("--task", type=int, help="run one task (an array index); default: all")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--out", type=Path, default=config.TRANSFER_FILE, help="merged transfer table")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.what == "spectra":
        return spectra(args.lmax)
    task_dir = args.out.with_name(args.out.stem + "_tasks")
    if args.merge:
        return merge(args.ells, args.n_l_chunks, task_dir, args.out, args.overwrite)
    if args.task is None and args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite to replace it")
    import camb
    # Same parameters as the stored table. Background and thermal history only:
    # T_source is evolved per k by get_time_evolution.
    results = camb.get_background(camb_params(max(2000, int(args.ells[-1]))))
    n_tasks = len(tasks(len(k_grid(float(results.conformal_time(0)))), args.ells, args.n_l_chunks))
    for task in ([args.task] if args.task is not None else range(n_tasks)):
        run_task(results, task, args.ells, args.n_l_chunks, task_dir, camb_version=camb.__version__)
    if args.task is None:
        merge(args.ells, args.n_l_chunks, task_dir, args.out, args.overwrite)


if __name__ == "__main__":
    main()
