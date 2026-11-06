"""Factorisable Fisher of the contact template (Smith, Senatore & Zaldarriaga 2015).

Manuscript Eqs. (cZ_odd) and (Fisher_po), full sky:
    F(L) = P sum_mu w_mu sum_{A,B} sum_{q in A, q' in B} w_q w_q' sum_{u in A, v in B} c_u c_v
           sum_{pi in S4} prod_a Z_{u_a, v_pi(a)}(x_q, x_q', mu),
    Z_{i j}(x, x', mu) = sum_{l = LMIN}^{L} (2l+1)/(4 pi) d^l_{s_i s_j}(mu) U_i,l(x) U_j,l(x'),
    U_i,l(x) = legs[i, l, x] / sqrt(C_l + N_l),
with A, B the radial families and P = config.fisher_prefactor(). Terms with the
same unordered legs are merged, since the full-band maps commute.

Gauss-Legendre with 2 L_max + 1 nodes is exact for every cutoff L <= L_max, so
one pass gives the whole F(L) curve: Z is accumulated in bands of l and
contracted after each band. F is additive over mu nodes, so disjoint node sets
can run as separate jobs. Only the unprojected Z is used: zeta vanishes for even
sum l, so F_odd = F_unprojected. parity_weight=True inserts (-1)^l into Z for
the check F_odd = (F - F_P)/2.
"""
from itertools import permutations
from math import lgamma

import numpy as np

import config
import terms


def wigner_d(lmax, mu):
    """d^l_{s s'}(mu) for s, s' in {-1, 0, 1} and l = 0..lmax, indexed [s+1, s'+1, mu, l].

    Three-term recursion in l (Smith et al. 2015, Eq. A3), seeded at
    l = max(|s|, |s'|).
    """
    mu = np.clip(np.atleast_1d(np.asarray(mu, dtype=float)), -1.0, 1.0)
    d = np.zeros((3, 3, len(mu), lmax + 1))
    for s in (-1, 0, 1):
        for sp in (-1, 0, 1):
            out = d[s + 1, sp + 1]
            lo = max(abs(s), abs(sp))
            # Seed: map (s, s') to s' >= |s| with the symmetries of d, then closed form.
            sign, a, b = 1.0, s, sp
            if b < 0 or (b == 0 and a < 0):
                sign, a, b = sign * (-1.0) ** (a - b), -a, -b
            if b < abs(a):
                sign, a, b = sign * (-1.0) ** (a - b), b, a
            pref = np.exp(0.5 * (lgamma(2 * b + 1) - lgamma(b - abs(a) + 1) - lgamma(b + abs(a) + 1)))
            out[:, lo] = (sign * (-1.0) ** (b - a) * pref
                          * ((1 + mu) / 2) ** ((b + a) / 2) * ((1 - mu) / 2) ** ((b - a) / 2))
            for ell in range(lo, lmax):
                alpha = np.sqrt((ell**2 - s**2) * (ell**2 - sp**2)) / ell if ell else 0.0
                beta = (2 * ell + 1) * (mu - (s * sp / (ell * (ell + 1)) if ell else 0.0))
                alpha_next = np.sqrt(((ell + 1) ** 2 - s**2) * ((ell + 1) ** 2 - sp**2)) / (ell + 1)
                previous = out[:, ell - 1] if ell else 0.0
                out[:, ell + 1] = (beta * out[:, ell] - alpha * previous) / alpha_next
    return d


def build_z(Ua, Ub, spins, D):
    """Z[i, j, q, q'] for one band of l; Ua, Ub are U at the two node blocks, D[s+1, s'+1, l]."""
    z = np.empty((len(spins), len(spins), Ua.shape[2], Ub.shape[2]))
    groups = {s: np.flatnonzero(spins == s) for s in (-1, 0, 1)}
    for s, I in groups.items():
        for sp, J in groups.items():
            block = np.tensordot(Ua[I] * D[s + 1, sp + 1][None, :, None], Ub[J], axes=([1], [1]))
            z[I[:, None], J[None, :]] = block.transpose(0, 2, 1, 3)
    return z


def contract(z, left, right, wa, wb):
    """sum_{u,v} c_u c_v sum_q sum_q' w_q w_q' sum_pi prod_a Z[u_a, v_pi(a)]."""
    (ca, ia), (cb, ib) = left, right
    coefficients = ca[:, None] * cb[None, :]
    total = 0.0
    for pi in permutations(range(4)):
        product = z[ia[:, None, 0], ib[None, :, pi[0]]]
        for a in range(1, 4):
            product = product * z[ia[:, None, a], ib[None, :, pi[a]]]
        total += np.einsum("uv,uvij,i,j->", coefficients, product, wa, wb, optimize=True)
    return total


def gauss_legendre(lmax):
    """The 2 lmax + 1 nodes that integrate the four-Z product exactly for every L <= lmax."""
    return np.polynomial.legendre.leggauss(2 * lmax + 1)


def fisher_factorisable(model, cl_total, cutoffs, nodes=None, x_block=16, parity_weight=False):
    """Full-sky F at each cutoff L, from the radial model of kernels.radial_model.

    nodes: indices into gauss_legendre(max(cutoffs)) to include (default all);
    results from disjoint node sets add up to the full F.
    """
    cutoffs = sorted({int(L) for L in cutoffs})
    lmax = cutoffs[-1]
    if cutoffs[0] < config.LMIN or model["legs"].shape[1] <= lmax:
        raise ValueError("cutoffs must lie in LMIN .. the kernel lmax")
    inverse_sqrt = np.zeros(lmax + 1)
    inverse_sqrt[config.LMIN:] = 1.0 / np.sqrt(cl_total[config.LMIN:lmax + 1])
    U = model["legs"][:, :lmax + 1] * inverse_sqrt[None, :, None]
    families = [(terms.merge(f["coeff"], f["legs"]), f["q"], f["w"]) for f in model["families"]]
    edges = [config.LMIN] + [L + 1 for L in cutoffs]
    bands = [np.arange(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]

    mu, w_mu = gauss_legendre(lmax)
    if nodes is not None:
        mu, w_mu = mu[nodes], w_mu[nodes]
    ell = np.arange(lmax + 1)
    weight = (2 * ell + 1) / (4 * np.pi) * ((-1.0) ** ell if parity_weight else 1.0)
    fisher = np.zeros(len(cutoffs))
    for m, (d, w) in enumerate(zip(wigner_d(lmax, mu).transpose(2, 0, 1, 3), w_mu)):
        D = d * weight
        for left, qa_all, wa_all in families:
            for right, qb_all, wb_all in families:
                for i in range(0, len(qa_all), x_block):
                    qa, wa = qa_all[i:i + x_block], wa_all[i:i + x_block]
                    for j in range(0, len(qb_all), x_block):
                        qb, wb = qb_all[j:j + x_block], wb_all[j:j + x_block]
                        z = 0.0
                        for k, band in enumerate(bands):
                            z = z + build_z(U[:, band][:, :, qa], U[:, band][:, :, qb],
                                            model["spins"], D[:, :, band])
                            fisher[k] += w * contract(z, left, right, wa, wb)
    return config.fisher_prefactor() * fisher
