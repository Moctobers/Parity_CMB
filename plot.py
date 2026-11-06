"""Figures: kernels over the noise, the reduced-trispectrum ladder, and SNR curves.

    python plot.py kernels --lmax 20 --sources sw transfer [--dimensionless] [--x 13900]
    python plot.py zeta --lmax 20 --sources sw transfer [--index-range 0 400]
    python plot.py snr --lmax 20 --curves direct:sw direct:transfer fisher:sw fisher:transfer

"transfer" reads data/kernels_lmax<lmax>.npz (python run.py kernels --lmax <lmax>); kernels
are shown at the radial node closest to --x (default chi_*). The p = 0 leg is left out of
the kernel plot: in the SW limit it is a delta shell, not a function of l. SNR curves are
read from the run.py outputs output/<method>_<source>_lmax<lmax>.npz. Figures go to output/.
"""
import argparse
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import kernels
import trispectrum
from run import load_model

COLORS = ("#2a78d6", "#eb6834", "#1baf7a")            # blue, orange, aqua
SOURCE_COLOR = {"sw": COLORS[0], "transfer": COLORS[1]}
NAME = {"sw": "analytic SW", "transfer": "CAMB transfer", "direct": "direct", "fisher": "factorisable"}
SOURCE_STYLE = {"sw": dict(ls="-", marker="o", ms=4), "transfer": dict(ls="--", marker="s", ms=5, mfc="none")}
METHOD_STYLE = {"direct": dict(ls="-", marker="o", ms=5), "fisher": dict(ls="--", marker="s", ms=6, mfc="none")}


def save(fig, name):
    path = config.OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


def kernel_values(source, lmax, x_eval):
    """B, dB of shape (4, lmax+1) at the evaluation radius, and that radius."""
    if source == "sw":
        B, dB = kernels.sw_kernels(lmax, config.chi_star())
        return B[:, :, 0], dB[:, :, 0], config.chi_star()
    path = config.DATA / f"kernels_lmax{lmax}.npz"
    if not path.exists():
        raise SystemExit(f"{path} is missing; make it with: python run.py kernels --lmax {lmax}")
    with np.load(path) as f:
        q = int(np.argmin(np.abs(f["x"] - x_eval)))
        return f["B"][:, :lmax + 1, q], f["dB"][:, :lmax + 1, q], float(f["x"][q])


def plot_kernels(args):
    chi, ell = config.chi_star(), np.arange(config.LMIN, args.lmax + 1)
    cl = config.total_spectrum(args.lmax)[ell]
    smooth = [i for i, p in enumerate(config.POWERS) if abs(p) > 1e-12]
    fig = plt.figure(figsize=(11.5, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.13])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    values = {source: kernel_values(source, args.lmax, args.x or chi) for source in args.sources}
    where = [f"{NAME[source]} at $x = {x:.1f}$ Mpc" for source, (_, _, x) in values.items()]
    for color, i in zip(COLORS, smooth):              # colour = power, line style = source
        p = config.POWERS[i]
        scale_b, scale_db = (chi ** (p + 3), chi ** (p + 4)) if args.dimensionless else (1.0, 1.0)
        for source, (B, dB, _) in values.items():
            label = f"{NAME[source]}, $p = {p:.2f}$"
            axes[0].plot(ell, (scale_b * B[i, ell]) ** 2 / cl, color=color, label=label, **SOURCE_STYLE[source])
            axes[1].plot(ell, (scale_db * dB[i, ell]) ** 2 / cl, color=color, label=label, **SOURCE_STYLE[source])
    legend = fig.add_subplot(grid[1, :])
    legend.axis("off")
    legend.legend(*axes[0].get_legend_handles_labels(), loc="center", ncol=3, frameon=False, fontsize=10)
    unit = (r"[\chi_*^{p+3}\hat B^{p}_\ell]^2", r"[\chi_*^{p+4}\partial_x\hat B^{p}_\ell]^2") if args.dimensionless \
        else (r"[\hat B^{p}_\ell]^2", r"[\partial_x\hat B^{p}_\ell]^2")
    for ax, numerator in zip(axes, unit):
        ax.set(yscale="log", xlabel=r"$\ell$", ylabel=rf"${numerator}/(C_\ell+N_\ell)$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    fig.suptitle("Kernels over the total TT variance: " + "; ".join(where), fontsize=11)
    tag = "_dimensionless" if args.dimensionless else ""
    save(fig, f"kernels_over_noise_{'_'.join(args.sources)}_lmax{args.lmax}{tag}.png")


def ordered_ladder(model, lmax):
    """zeta over l1 < l2 < l3 < l4 <= lmax with odd sum, each with its full L range."""
    prepared, rows, values = trispectrum.prepare(model), [], []
    for quad in combinations(range(config.LMIN, lmax + 1), 4):
        l1, l2, l3, l4 = quad
        lo, hi = max(abs(l1 - l2), abs(l3 - l4)), min(l1 + l2, l3 + l4)
        if sum(quad) % 2 == 0 or lo > hi:
            continue
        L = np.arange(lo, hi + 1)
        rows += [(*quad, star) for star in L]
        values.append(trispectrum.zeta(model, quad, L, prepared))
    return np.array(rows), np.concatenate(values)


def symlog_ticks(linthresh, vmax, max_decades=5):
    decades = list(range(int(np.ceil(np.log10(linthresh))), int(np.floor(np.log10(vmax))) + 1))
    while len(decades) > max_decades:
        decades = decades[::2]
    return [-(10.0 ** d) for d in reversed(decades)] + [0.0] + [10.0 ** d for d in decades]


def plot_zeta(args):
    curves = {}
    for source in args.sources:
        configs, curves[source] = ordered_ladder(load_model(source, args.lmax)[0], args.lmax)
    lo, hi = args.index_range or (0, len(configs))
    index, configs = np.arange(lo, hi), configs[lo:hi]
    dense = len(index) > 400
    fig, axes = plt.subplots(6, 1, figsize=(max(11.0, min(18.0, 9.0 + len(index) / 1600)), 12.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1]}, constrained_layout=True)
    peak = max(np.abs(z[lo:hi]).max() for z in curves.values())
    for source, z in curves.items():                  # later sources are drawn on top
        style = dict(SOURCE_STYLE[source], marker=None if dense else SOURCE_STYLE[source]["marker"])
        axes[0].plot(index, z[lo:hi], color=SOURCE_COLOR[source], lw=0.9 if dense else 1.4,
                     alpha=0.8 if dense else 1.0, label=NAME[source], **style)
    axes[0].set_yscale("symlog", linthresh=peak * 1e-6)
    axes[0].set_yticks(symlog_ticks(peak * 1e-6, peak))
    axes[0].set_ylim(-4 * peak, 4 * peak)
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[0].set_ylabel(r"$\zeta^{\ell_1\ell_2}_{\ell_3\ell_4}(\ell_*)$", fontsize=18)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(curves), frameon=False, fontsize=12)
    fig.suptitle(rf"Reduced trispectrum, $\ell_1<\ell_2<\ell_3<\ell_4\leq{args.lmax}$, odd $\sum\ell$, "
                 rf"all triangle-allowed $\ell_*$ ({len(index)} configurations)", fontsize=13)
    upper = {4: int(configs[:, 4].max())}             # l_* scaled to the window; the external l to lmax
    for ax, column, label in zip(axes[1:], (4, 0, 1, 2, 3), (r"$\ell_*$", r"$\ell_1$", r"$\ell_2$", r"$\ell_3$", r"$\ell_4$")):
        ax.plot(index, configs[:, column], color="#37474f", lw=0.7 if dense else 1.2,
                drawstyle="default" if dense else "steps-mid")
        ax.set_ylabel(label, fontsize=18)
        ax.set_ylim(0 if column == 4 else config.LMIN - 1, upper.get(column, args.lmax) + 1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"configuration index (ordered by $\ell_1, \ell_2, \ell_3, \ell_4, \ell_*$)", fontsize=16)
    axes[-1].set_xlim(index[0] - 0.5, index[-1] + 0.5)
    tag = f"_idx{lo}-{hi}" if args.index_range else ""
    save(fig, f"zeta_ladder_{'_'.join(args.sources)}_lmax{args.lmax}{tag}.png")


def plot_snr(args):
    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    f_sky = set()
    for curve in args.curves:
        method, source = curve.split(":")
        path = config.OUTPUT / f"{method}_{source}_lmax{args.lmax}.npz"
        if not path.exists():
            raise SystemExit(f"{path} is missing; make it with: python run.py {method} --source {source} "
                             f"--lmax {args.lmax}")
        with np.load(path) as f:
            L, snr = f["L"], f["snr"]
            f_sky.add(float(f["f_sky"]))
        ax.plot(L, snr, color=SOURCE_COLOR[source], label=f"{NAME[method]}, {NAME[source]}: {snr[-1]:.4g}",
                **METHOD_STYLE[method])
    if len(f_sky) > 1:
        raise SystemExit(f"the curves use different sky fractions: {sorted(f_sky)}")
    ax.set_xlabel(r"$\ell_{\max}$", fontsize=14)
    ax.set_ylabel(r"cumulative SNR$(\ell_{\max})$", fontsize=14)
    ax.set_title(rf"Parity-odd contact template, $f_{{\rm sky}} = {f_sky.pop():g}$, $C_\ell+N_\ell$ noise")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)
    save(fig, f"snr_lmax{args.lmax}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=("kernels", "zeta", "snr"))
    parser.add_argument("--lmax", type=int, required=True)
    parser.add_argument("--sources", nargs="+", choices=("sw", "transfer"), default=["sw"])
    parser.add_argument("--curves", nargs="+", default=["direct:sw", "fisher:sw"],
                        help="method:source pairs, method in {direct, fisher}")
    parser.add_argument("--dimensionless", action="store_true", help="kernels: scale B by chi*^(p+3)")
    parser.add_argument("--x", type=float, help="kernels: radius in Mpc for the transfer kernels")
    parser.add_argument("--index-range", type=int, nargs=2, metavar=("LO", "HI"), help="zeta: index window")
    args = parser.parse_args()
    {"kernels": plot_kernels, "zeta": plot_zeta, "snr": plot_snr}[args.figure](args)


if __name__ == "__main__":
    main()
