"""Radial kernels, the radial grid, and the radial model shared by zeta and the Fisher.

Phase-stripped kernels (the i^l of manuscript Eq. BpGi goes into the zeta prefactor):
    B^p_l(x)  = 1/(2 pi^2) int dk k^(p+2) j_l(kx)  Delta_l(k),
    dB^p_l(x) = 1/(2 pi^2) int dk k^(p+3) j'_l(kx) Delta_l(k).
In the Sachs-Wolfe limit Delta_l = -j_l(k chi)/5, the p = 0 kernel is a shell,
B^0_l(x) = -delta(x - chi)/(20 pi x chi), so every radial integral collapses to
x = chi with weight -1/(20 pi chi^2) and no IBP boundary terms. The other
kernels at x = chi are closed form (Gradshteyn 6.574) and dB = -(p+3) B/(2 chi).
"""
import numpy as np
from scipy.special import gammaln, spherical_jn

import config
import terms


def sw_kernels(lmax, chi, powers=config.POWERS):
    """B, dB at x = chi with shape (4, lmax+1, 1); the p = 0 row holds the shell weight."""
    B, dB = np.zeros((4, lmax + 1)), np.zeros((4, lmax + 1))
    ell = np.arange(config.LMIN, lmax + 1)
    for i, p in enumerate(powers):
        if abs(p) < 1e-12:
            B[i, config.LMIN:] = -1.0 / (20.0 * np.pi * chi**2)
            continue
        log_b = (np.log(1 / (10 * np.pi)) + p * np.log(2) - (p + 3) * np.log(chi)
                 + gammaln(-(p + 1)) - 2 * gammaln(-p / 2)
                 + gammaln(ell + (p + 3) / 2) - gammaln(ell + (1 - p) / 2))
        B[i, config.LMIN:] = -np.exp(log_b)
        dB[i] = -(p + 3) * B[i] / (2 * chi)
    return B[..., None], dB[..., None]


def load_transfer(path=config.TRANSFER_FILE):
    """Delta_l(k) as a dense table T[l, k], with the k grid, the stored rows and eta0."""
    with np.load(path) as f:
        if not bool(f["complete"][0]):
            raise RuntimeError(f"{path} is marked incomplete")
        stored = dict(zip(f["cosmology_parameter_names"], f["cosmology_parameter_values"]))
        expected = dict(H0=config.H0, ombh2=config.OMBH2, omch2=config.OMCH2,
                        tau=config.TAU, A_s=config.A_S, n_s=config.N_S)
        if any(stored[name] != value for name, value in expected.items()):
            raise RuntimeError(f"{path} was made with a different cosmology")
        ells = f["ells_sparse"].astype(int)
        T = np.zeros((ells.max() + 1, len(f["k_arr"])))
        T[ells] = f["Tgamma_lk"]
        return dict(T=T, k=f["k_arr"].copy(), ells=ells, eta0=float(f["eta0"][0]))


def trapezoid_weights(x):
    w = np.zeros(len(x))
    w[:-1] += np.diff(x) / 2
    w[1:] += np.diff(x) / 2
    return w


def transfer_kernels(table, x, ells, powers=config.POWERS, x_block=64):
    """B, dB with shape (4, len(ells), len(x)), by the trapezoid rule in k."""
    k = table["k"]
    wk = trapezoid_weights(k) / (2 * np.pi**2)
    B, dB = np.zeros((4, len(ells), len(x))), np.zeros((4, len(ells), len(x)))
    for j, ell in enumerate(ells):
        if ell not in table["ells"]:
            raise ValueError(f"the transfer table has no row for l = {ell}")
        row = wk * table["T"][ell]
        w_b = np.array([row * k ** (p + 2) for p in powers])
        w_db = np.array([row * k ** (p + 3) for p in powers])
        for s in range(0, len(x), x_block):
            arg = np.outer(k, x[s:s + x_block])
            B[:, j, s:s + x_block] = w_b @ spherical_jn(ell, arg)
            dB[:, j, s:s + x_block] = w_db @ spherical_jn(ell, arg, derivative=True)
    return B, dB


def radial_grid(spec, chi, eta0):
    """Nodes from "start:stop:step" segments; "chi" and "eta0" may carry a +/- offset in Mpc."""
    def value(text):
        text = text.strip()
        for name, v in (("eta0", eta0), ("chi", chi)):
            if text.startswith(name):
                return v + float(text[len(name):] or 0.0)
        return float(text)

    pieces, end = [], None
    for segment in spec.split(","):
        start, stop, step = map(value, segment.split(":"))
        if end is not None and start != end:
            raise ValueError(f"radial segments must be contiguous: {spec}")
        nodes = start + step * np.arange(int(np.floor((stop - start) / step)) + 1)
        nodes = np.append(nodes[nodes < stop], stop)
        pieces.append(nodes if end is None else nodes[1:])
        end = stop
    x = np.concatenate(pieces)
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"radial grid is not increasing: {spec}")
    return x


def radial_model(B, dB, x, powers=config.POWERS):
    """Leg arrays and signed term families from kernels of shape (4, lmax+1, n_x).

    One node (the SW shell): the bulk terms with weight 1. Otherwise: the bulk
    terms with trapezoid weights, and the IBP boundary terms at x_min (weight -1)
    and x_max (weight +1). legs[i, l, q] = gamma_l^s [B or dB]^p_l(x_q) for label
    i = (p, kind, s), with gamma^0 = 1 and gamma^{+-1} = +-sqrt(l(l+1)).
    """
    bulk, boundary = terms.contact_terms(powers)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if len(x) == 1:
        families = [(bulk, [0], [1.0])]
    else:
        families = [(bulk, np.arange(len(x)), trapezoid_weights(x)),
                    (boundary, [0], [-1.0]), (boundary, [len(x) - 1], [1.0])]
    labels, encoded = terms.encode([f[0] for f in families])
    ell = np.arange(B.shape[1])
    gamma = {0: np.ones(len(ell)), 1: np.sqrt(ell * (ell + 1.0)), -1: -np.sqrt(ell * (ell + 1.0))}
    legs = np.array([gamma[s][:, None] * (B if kind == "B" else dB)[p] for p, kind, s in labels])
    return dict(x=x, labels=labels, spins=np.array([s for _, _, s in labels]), legs=legs,
                families=[dict(coeff=c, legs=i, q=np.asarray(q), w=np.asarray(w, dtype=float))
                          for (c, i), (_, q, w) in zip(encoded, families)])


def sw_model(lmax, chi=None):
    chi = config.chi_star() if chi is None else chi
    return radial_model(*sw_kernels(lmax, chi), [chi])
