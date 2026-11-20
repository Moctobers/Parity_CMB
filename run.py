"""Kernels, the reduced trispectrum, and the two Fisher calculations.

    python run.py kernels --lmax 20                                  # on config.RADIAL_GRID
    python run.py kernels --lmax 2000 --n-chunks 64 --task 3         # one l chunk (array task)
    python run.py kernels --lmax 2000 --n-chunks 64 --merge
    python run.py direct --source sw --lmax 20 [--save-grid]
    python run.py direct --source transfer --lmax 20
    python run.py fisher --source sw --lmax 100 --ell-step 10
    python run.py fisher --source transfer --lmax 2000 --n-chunks 400 --task 7   # one mu chunk
    python run.py fisher --source transfer --lmax 2000 --n-chunks 400 --merge

F is full sky. Every output also stores SNR = sqrt(f_sky F) for --f-sky (default: the experiment's).
--experiment (default config.EXPERIMENT) sets the noise; it is part of the direct/fisher output names,
e.g. output/fisher_transfer_so_lmax2000.npz, and is stored in the file with its noise channels.
Transfer kernels are read from --kernels (default data/kernels_lmax<lmax>.npz).
"""
import argparse
import time
from pathlib import Path

import numpy as np

import config
import kernels
import trispectrum
from fisher_direct import fisher_direct
from fisher_factorisable import fisher_factorisable, gauss_legendre


def chunk_dir(path):
    return path.with_name(path.stem + "_chunks")


def merge_chunks(path, n_chunks, add):
    files = sorted(chunk_dir(path).glob("*.npz"))
    if len(files) != n_chunks:
        raise RuntimeError(f"found {len(files)} of {n_chunks} chunk files in {chunk_dir(path)}")
    parts = [dict(np.load(f)) for f in files]
    return add(parts)


def kernels_command(args):
    out = args.out or config.DATA / f"kernels_lmax{args.lmax}.npz"
    table = kernels.load_transfer()
    x = kernels.radial_grid(args.grid, config.chi_star(), table["eta0"])
    meta = dict(x=x, grid=args.grid, lmax=args.lmax, powers=np.array(config.POWERS),
                transfer_file=Path(config.TRANSFER_FILE).name)
    if args.merge:
        def add(parts):
            B = np.zeros((4, args.lmax + 1, len(x)))
            dB = np.zeros_like(B)
            for p in parts:
                B[:, p["ells"]], dB[:, p["ells"]] = p["B"], p["dB"]
            return B, dB
        B, dB = merge_chunks(out, args.n_chunks, add)
        np.savez(out, B=B, dB=dB, **meta)
        return print(f"saved {out}")
    ells = np.arange(config.LMIN, args.lmax + 1)
    if args.task is not None:
        ells = np.array_split(ells, args.n_chunks)[args.task]
    t = time.time()
    B, dB = kernels.transfer_kernels(table, x, ells)
    if args.task is not None:
        path = chunk_dir(out) / f"{args.task:05d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, B=B, dB=dB, ells=ells, **meta)
    else:
        full_B, full_dB = np.zeros((4, args.lmax + 1, len(x))), np.zeros((4, args.lmax + 1, len(x)))
        full_B[:, ells], full_dB[:, ells] = B, dB
        path = out
        np.savez(path, B=full_B, dB=full_dB, **meta)
    print(f"saved {path}: l = {ells[0]}..{ells[-1]}, {len(x)} radial nodes, {time.time() - t:.0f} s")


def load_model(source, lmax, path=None):
    """Radial model for "sw" or "transfer" (kernels from path), and its radial-grid label."""
    if source == "sw":
        return kernels.sw_model(lmax), "sw"
    path = Path(path or config.DATA / f"kernels_lmax{lmax}.npz")
    if not path.exists():
        raise SystemExit(f"{path} is missing; make it with: python run.py kernels --lmax {lmax}")
    with np.load(path) as f:
        if int(f["lmax"]) < lmax:
            raise SystemExit(f"{path} only reaches l = {int(f['lmax'])}")
        if "powers" not in f.files or not np.allclose(f["powers"], config.POWERS):
            raise SystemExit(f"{path} was made for other template powers than config.POWERS; "
                             f"remake it with: python run.py kernels --lmax {lmax}")
        model = kernels.radial_model(f["B"][:, :lmax + 1], f["dB"][:, :lmax + 1], f["x"])
        return model, str(f["grid"])


def noise_meta(args):
    """The experiment and its (uK arcmin, FWHM arcmin) channels, stored with every result."""
    return dict(experiment=args.experiment,
                noise_channels=np.array(config.EXPERIMENTS[args.experiment]["channels"]))


def save(path, F, f_sky, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, F=F, snr=np.sqrt(f_sky * np.maximum(F, 0)), f_sky=f_sky, **extra)
    print(f"saved {path}; SNR(lmax) = {np.sqrt(f_sky * max(F[-1], 0)):.6g} at f_sky = {f_sky}")


def direct_command(args):
    model, grid_spec = load_model(args.source, args.lmax, args.kernels)
    t = time.time()
    grid = trispectrum.grid(model, args.lmax)
    L, F = fisher_direct(grid, config.total_spectrum(args.lmax, args.experiment))
    print(f"{len(grid['zeta'])} odd configurations in {time.time() - t:.0f} s")
    out = args.out or config.OUTPUT / f"direct_{args.source}_{args.experiment}_lmax{args.lmax}.npz"
    save(out, F, args.f_sky, L=L, grid_spec=grid_spec, **noise_meta(args), **(grid if args.save_grid else {}))


def fisher_command(args):
    model, grid_spec = load_model(args.source, args.lmax, args.kernels)
    cutoffs = np.unique(np.r_[np.arange(args.ell_step, args.lmax, args.ell_step), args.lmax])
    cutoffs = cutoffs[cutoffs >= config.LMIN]
    out = args.out or config.OUTPUT / f"fisher_{args.source}_{args.experiment}_lmax{args.lmax}.npz"
    if args.merge:
        def add(parts):
            made = {str(p.get("experiment", "unknown")) for p in parts}
            if made != {args.experiment}:
                raise SystemExit(f"chunks were made for {sorted(made)}, not --experiment {args.experiment}")
            return sum(p["F_partial"] for p in parts)
        F = merge_chunks(out, args.n_chunks, add)
        return save(out, F, args.f_sky, L=cutoffs, grid_spec=grid_spec, **noise_meta(args))
    nodes = np.arange(len(gauss_legendre(args.lmax)[0]))
    if args.task is not None:
        nodes = np.array_split(nodes, args.n_chunks)[args.task]
    t = time.time()
    F = fisher_factorisable(model, config.total_spectrum(args.lmax, args.experiment), cutoffs, nodes=nodes,
                            x_block=args.x_block)
    print(f"{len(nodes)} mu nodes in {time.time() - t:.0f} s")
    if args.task is None:
        return save(out, F, args.f_sky, L=cutoffs, grid_spec=grid_spec, **noise_meta(args))
    path = chunk_dir(out) / f"{args.task:05d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, F_partial=F, L=cutoffs, nodes=nodes, experiment=args.experiment)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("kernels", "direct", "fisher"))
    parser.add_argument("--lmax", type=int, required=True)
    parser.add_argument("--source", choices=("sw", "transfer"), default="transfer")
    parser.add_argument("--grid", default=config.RADIAL_GRID, help="radial grid segments (kernels)")
    parser.add_argument("--kernels", help="kernel file for --source transfer")
    parser.add_argument("--ell-step", type=int, default=1, help="spacing of the F(L) cutoffs")
    parser.add_argument("--experiment", choices=tuple(config.EXPERIMENTS), default=config.EXPERIMENT,
                        help="noise model (direct, fisher)")
    parser.add_argument("--f-sky", type=float, help="default: the experiment's f_sky")
    parser.add_argument("--x-block", type=int, default=16, help="radial nodes per Z block (fisher)")
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--task", type=int, help="chunk index for an array job")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--save-grid", action="store_true", help="store zeta in the direct output")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.f_sky is None:
        args.f_sky = config.EXPERIMENTS[args.experiment]["f_sky"]
    {"kernels": kernels_command, "direct": direct_command, "fisher": fisher_command}[args.command](args)


if __name__ == "__main__":
    main()
