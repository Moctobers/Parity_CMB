"""Direct Fisher (= SNR^2 for one amplitude) from a complete grid of zeta.

Manuscript Eq. (SNR_cmb), full sky:
    F(L) = 1/4! sum_{l_a = LMIN}^{L} sum_{L*} |zeta|^2 / prod_a (C_{l_a} + N_{l_a}),
with all orderings of (l1..l4) and odd sum l only (the parity projector).
"""
import numpy as np

import config


def fisher_direct(grid, cl_total, lmax=None):
    """F(L) for L = LMIN .. lmax from trispectrum.grid output; returns (L, F)."""
    lmax = grid["lmax"] if lmax is None else lmax
    if grid["lmin"] != config.LMIN or lmax > grid["lmax"]:
        raise ValueError("the grid does not cover LMIN .. lmax")
    ells = grid["ells"].astype(int)
    largest = ells.max(axis=1)
    keep = largest <= lmax
    contribution = grid["zeta"][keep] ** 2 / (24 * np.prod(cl_total[ells[keep]], axis=1))
    increment = np.bincount(largest[keep], weights=contribution, minlength=lmax + 1)
    return np.arange(config.LMIN, lmax + 1), np.cumsum(increment[config.LMIN:])
