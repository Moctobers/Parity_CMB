"""Reduced trispectrum zeta^{l1 l2}_{l3 l4}(L) of the parity-odd contact template.

Manuscript Eqs. (alm_sw) and (reduced_trispecx), extended to the full transfer:
    zeta = (i/2) i^(sum l) A sqrt(2L+1)/(4 pi) prod_a sqrt(2 l_a + 1)
           x sum_u c_u I_u (-1)^(L + s3) (l1 l2 L; -s1 -s2 -s3) (L l3 l4; s3 -s3 0),
    I_u  = sum_q w_q prod_a legs[u_a, l_a, q]      (sum over the radial families).
A is config.zeta_amplitude(). zeta vanishes for even sum l and is real for odd sum l.
"""
from functools import lru_cache
from itertools import permutations, product

import numpy as np

import config

SPIN_TUPLES = list(permutations((0, 1, -1)))


@lru_cache(maxsize=None)
def three_j(l2, l3, m2, m3):
    """(l1 l2 l3; -m2-m3 m2 m3) for l1 = 0 .. l2+l3, zero where not allowed.

    Three-term recursion in l1 (Schulten & Gordon 1975), run up from l1_min and
    down from l1_max and matched in the middle; normalised by
    sum_l1 (2 l1 + 1) f^2 = 1, with sign (-1)^(l2 - l3 - m1) at l1 = l2 + l3.
    """
    m1 = -m2 - m3
    out = np.zeros(l2 + l3 + 1)
    lo, hi = max(abs(l2 - l3), abs(m1)), l2 + l3
    if abs(m2) > l2 or abs(m3) > l3 or lo > hi:
        return out
    j = np.arange(lo, hi + 1, dtype=float)
    # j X(j+1) f(j+1) + Y(j) f(j) + (j+1) X(j) f(j-1) = 0
    X = np.sqrt(np.maximum((j**2 - (l2 - l3) ** 2) * ((l2 + l3 + 1) ** 2 - j**2) * (j**2 - m1**2), 0))
    X = np.append(X, 0.0)
    Y = -(2 * j + 1) * ((l2 * (l2 + 1) - l3 * (l3 + 1)) * m1 - j * (j + 1) * (m3 - m2))
    n = len(j)
    down = np.zeros(n + 1)
    down[n - 1] = 1.0
    for i in range(n - 1, 0, -1):
        down[i - 1] = -(Y[i] * down[i] + j[i] * X[i + 1] * down[i + 1]) / ((j[i] + 1) * X[i])
    f = down[:n]
    if lo > 0 and n > 2:                      # upward recursion is undefined from l1 = 0
        up = np.zeros(n)
        up[0] = 1.0
        for i in range(n - 1):
            up[i + 1] = -(Y[i] * up[i] + (j[i] + 1) * X[i] * (up[i - 1] if i else 0)) / (j[i] * X[i + 1])
        mid = n // 2
        window = slice(max(mid - 2, 0), mid + 3)
        scale = up[window] @ f[window] / (up[window] @ up[window])
        f = np.concatenate([scale * up[:mid], f[mid:]])
    f = f / np.sqrt(np.sum((2 * j + 1) * f**2))
    out[lo:] = f * np.sign(f[-1]) * (-1) ** (l2 - l3 - m1)
    return out


def prepare(model):
    """Per family: legs at its nodes, term labels, coefficients, weights, spin-tuple index."""
    spins = model["spins"]
    return [(model["legs"][:, :, fam["q"]], fam["legs"], fam["coeff"], fam["w"],
             np.array([SPIN_TUPLES.index(tuple(spins[row[:3]])) for row in fam["legs"]]))
            for fam in model["families"]]


def zeta(model, ells, stars, prepared=None):
    """zeta(l1, l2, l3, l4; L) for the internal multipoles L in stars; sum l must be odd.

    Pass prepared = prepare(model) when calling this in a loop.
    """
    l1, l2, l3, l4 = map(int, ells)
    if (l1 + l2 + l3 + l4) % 2 == 0:
        raise ValueError("the parity-odd template vanishes for even sum l")
    stars = np.asarray(stars)
    weights = np.zeros(len(SPIN_TUPLES))
    for legs, idx, coeff, w, spin_id in prepared or prepare(model):
        radial = (legs[idx[:, 0], l1] * legs[idx[:, 1], l2]
                  * legs[idx[:, 2], l3] * legs[idx[:, 3], l4]) @ w
        weights += np.bincount(spin_id, weights=coeff * radial, minlength=len(SPIN_TUPLES))
    angular = np.array([(-1.0) ** (stars + s3) * three_j(l1, l2, -s1, -s2)[stars]
                        * three_j(l3, l4, -s3, 0)[stars] for s1, s2, s3 in SPIN_TUPLES])
    phase = (-1) ** ((l1 + l2 + l3 + l4 + 1) // 2) / 2      # (i/2) i^(sum l), real for odd sum
    norm = np.sqrt((2 * stars + 1) * (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) * (2 * l4 + 1))
    return phase * config.zeta_amplitude() * norm / (4 * np.pi) * (weights @ angular)


def grid(model, lmax, lmin=config.LMIN):
    """zeta on every ordered assignment l_a in [lmin, lmax] with odd sum, all allowed L."""
    prepared = prepare(model)
    ells, stars, values = [], [], []
    for quad in product(range(lmin, lmax + 1), repeat=4):
        l1, l2, l3, l4 = quad
        lo, hi = max(abs(l1 - l2), abs(l3 - l4)), min(l1 + l2, l3 + l4)
        if sum(quad) % 2 == 0 or lo > hi:
            continue
        L = np.arange(lo, hi + 1)
        ells.append(np.broadcast_to(quad, (len(L), 4)))
        stars.append(L)
        values.append(zeta(model, quad, L, prepared))
    return dict(ells=np.concatenate(ells).astype(np.int16), star=np.concatenate(stars).astype(np.int16),
                zeta=np.concatenate(values), lmin=lmin, lmax=lmax)
