"""Signed four-leg terms of the contact template after integration by parts.

A leg is (power index, kind, spin): kind "B" or "dB" (radial derivative), spin
0, +1 or -1 for (d_x, eth, ethbar). Before IBP there are 4! power assignments G
times 3! operator assignments sigma = 144 terms with sign(G) sign(sigma); legs
1-3 carry the operators and leg 4 none. In the 36 terms where d_x hits the
p = 0 leg, IBP gives
    int dx dB^0 F = [B^0 F]_{x_min}^{x_max} - int dx B^0 dF,
i.e. one boundary term (same sign) and three bulk terms with the derivative on
another leg (opposite sign). Total: 108 + 108 = 216 bulk and 36 boundary terms.
"""
from itertools import permutations

import numpy as np


def perm_sign(perm):
    inversions = sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm)))
    return -1 if inversions % 2 else 1


def contact_terms(powers):
    """Return (bulk, boundary) lists of (coefficient, (leg1, leg2, leg3, leg4))."""
    p0 = [i for i, p in enumerate(powers) if abs(p) < 1e-12]
    if len(powers) != 4 or len(p0) != 1:
        raise ValueError("the contact template needs four powers with exactly one p = 0")
    p0 = p0[0]
    rank = {0: 0, 1: 1, -1: 2}
    bulk, boundary = [], []
    for g in permutations(range(4)):
        for spins in permutations((0, 1, -1)):
            c = float(perm_sign(g) * perm_sign([rank[s] for s in spins]))
            legs = [(g[a], "dB" if spins[a] == 0 else "B", spins[a]) for a in range(3)]
            legs.append((g[3], "B", 0))
            a0 = g.index(p0)
            if legs[a0][1] == "B":
                bulk.append((c, tuple(legs)))
                continue
            legs[a0] = (p0, "B", 0)
            boundary.append((c, tuple(legs)))
            for a in range(4):
                if a != a0:
                    moved = list(legs)
                    moved[a] = (legs[a][0], "dB", legs[a][2])
                    bulk.append((-c, tuple(moved)))
    return bulk, boundary


def encode(families):
    """Turn term lists into a shared label table and integer arrays.

    families: list of term lists. Returns (labels, [(coeff, leg_index), ...]),
    with leg_index[u, a] the label of leg a in term u.
    """
    labels = sorted({leg for terms in families for _, legs in terms for leg in legs})
    index = {label: i for i, label in enumerate(labels)}
    encoded = [(np.array([c for c, _ in terms]),
                np.array([[index[leg] for leg in legs] for _, legs in terms], dtype=np.int64))
               for terms in families]
    return labels, encoded


def merge(coeff, leg_index):
    """Sum coefficients of terms with the same unordered set of legs.

    Valid only where the four legs commute, i.e. full-band maps in the
    factorisable Fisher. The trispectrum itself keeps the ordered legs.
    """
    rows, inverse = np.unique(np.sort(leg_index, axis=1), axis=0, return_inverse=True)
    summed = np.bincount(inverse.ravel(), weights=coeff, minlength=len(rows))
    keep = summed != 0
    return summed[keep], rows[keep]
