"""Checks of the core calculations.

    python -m pytest -q test_core.py          # about a minute
    FULL=1 python -m pytest -q test_core.py   # adds the l <= 20 references (a few minutes)

tests/reference.npz holds outputs of the old codes/ (see its "note"), made with
the C_l + N_l stored in it. Everything else is checked against exact results or
against the other method.
"""
import os
from fractions import Fraction
from itertools import product
from math import factorial, sqrt
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn

import camb_inputs
import config
import kernels
import terms
import trispectrum
from fisher_direct import fisher_direct
from fisher_factorisable import fisher_factorisable, wigner_d

REF = np.load(Path(__file__).parent / "tests" / "reference.npz")
FULL = os.environ.get("FULL") == "1"
CL = REF["cl_total"]


def relative(a, b):
    return np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b))


def test_terms():
    bulk, boundary = terms.contact_terms(config.POWERS)
    assert (len(bulk), len(boundary)) == (216, 36)
    p0 = config.POWERS.index(0.0)
    for _, legs in bulk + boundary:
        assert sorted(s for _, _, s in legs[:3]) == [-1, 0, 1] and legs[3][2] == 0
        assert sorted(p for p, _, _ in legs) == [0, 1, 2, 3]
        assert all(kind == "B" for p, kind, _ in legs if p == p0)       # no derivative on p = 0
    labels, ((cb, ib), (cs, is_)) = terms.encode([bulk, boundary])
    assert len(labels) == 21
    assert (len(terms.merge(cb, ib)[0]), len(terms.merge(cs, is_)[0])) == (30, 6)


def exact_three_j(j1, j2, j3, m1, m2, m3):
    """Racah formula in exact rational arithmetic."""
    if m1 + m2 + m3 or not abs(j1 - j2) <= j3 <= j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    f = factorial
    pre = Fraction(f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3), f(j1 + j2 + j3 + 1))
    pre *= f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3)
    s = sum(Fraction((-1) ** k, f(k) * f(j1 + j2 - j3 - k) * f(j1 - m1 - k) * f(j2 + m2 - k)
                     * f(j3 - j2 + m1 + k) * f(j3 - j1 - m2 + k))
            for k in range(max(0, j2 - j3 - m1, j1 - j3 + m2), min(j1 + j2 - j3, j1 - m1, j2 + m2) + 1))
    return (-1) ** (j1 - j2 - m3) * sqrt(float(pre * s * s)) * (1 if s > 0 else -1)


def test_three_j():
    for l2, l3, m2, m3 in product(range(13), range(13), (-1, 0, 1), (-1, 0, 1)):
        values = trispectrum.three_j(l2, l3, m2, m3)
        expected = [exact_three_j(l1, l2, l3, -m2 - m3, m2, m3) for l1 in range(l2 + l3 + 1)]
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-14)
    try:
        import pywigxjpf as wig
    except ImportError:
        return
    wig.wig_table_init(2 * 400, 3)
    wig.wig_temp_init(2 * 400)
    for l2, l3, m2, m3 in ((150, 170, 1, -1), (200, 199, 0, 1), (90, 3, -1, 0)):
        values = trispectrum.three_j(l2, l3, m2, m3)
        expected = [wig.wig3jj(2 * l1, 2 * l2, 2 * l3, -2 * (m2 + m3), 2 * m2, 2 * m3)
                    for l1 in range(l2 + l3 + 1)]
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-13)


def test_wigner_d_orthogonality():
    mu, w = np.polynomial.legendre.leggauss(64)
    d = wigner_d(30, mu)
    for s, sp in product(range(3), range(3)):
        lo = max(abs(s - 1), abs(sp - 1))
        gram = (d[s, sp].T * w) @ d[s, sp]
        np.testing.assert_allclose(gram[lo:, lo:], np.diag(2 / (2 * np.arange(lo, 31) + 1)), atol=1e-13)


def test_sw_kernels():
    # For p = -3 (n_s = 1): C_l = 4 pi (-1/5) (2 pi^2 A_s) B^{-3}_l(chi) = 2 pi A_s / (25 l (l+1)).
    B, _ = kernels.sw_kernels(20, config.chi_star(), powers=(-3.0, -4.0, -5.0, 0.0))
    ell = np.arange(2, 21)
    np.testing.assert_allclose(4 * np.pi * (-0.2) * 2 * np.pi**2 * config.A_S * B[0, 2:, 0],
                               2 * np.pi * config.A_S / (25 * ell * (ell + 1)), rtol=1e-13)
    # The numerical k-integrals reproduce the closed forms for Delta_l = -j_l(k chi)/5.
    chi, k = config.chi_star(), kernels.load_transfer()["k"]
    ells = np.arange(2, 11)
    T = np.zeros((11, len(k)))
    T[2:] = [-spherical_jn(ell, k * chi) / 5 for ell in ells]
    B, dB = kernels.transfer_kernels(dict(T=T, k=k, ells=ells), np.array([chi]), ells)
    Bs, dBs = kernels.sw_kernels(10, chi)
    np.testing.assert_allclose(B[:3, :, 0], Bs[:3, 2:, 0], rtol=5e-4)
    np.testing.assert_allclose(dB[:3, :, 0], dBs[:3, 2:, 0], rtol=5e-4)


def test_transfer_kernels_reference():
    x = REF["kernel_x"]
    assert np.array_equal(kernels.radial_grid(config.SMITH_GRID, config.chi_star(), 14150.094559439956), x)
    B, dB = kernels.transfer_kernels(kernels.load_transfer(), x, np.arange(2, 9))
    assert relative(B, REF["kernel_B"][:, 2:]) < 1e-12 and relative(dB, REF["kernel_dB"][:, 2:]) < 1e-12


def reference_models(lmax):
    return {"sw": kernels.sw_model(lmax),
            "camb": kernels.radial_model(REF["kernel_B"][:, :lmax + 1], REF["kernel_dB"][:, :lmax + 1],
                                         REF["kernel_x"])}


def test_zeta_and_direct_reference():
    for name, model in reference_models(8).items():
        grid = trispectrum.grid(model, 8)
        assert np.array_equal(grid["ells"], REF["ells"]) and np.array_equal(grid["star"], REF["star"])
        assert relative(grid["zeta"], REF[f"zeta_{name}"]) < 1e-12
        _, F = fisher_direct(grid, CL)
        assert relative(F, REF[f"F_direct_{name}"][:7]) < 1e-12


@pytest.mark.skipif(not FULL, reason="set FULL=1")
def test_direct_reference_lmax20():
    x = REF["kernel_x"]
    B, dB = np.zeros((4, 21, len(x))), np.zeros((4, 21, len(x)))
    B[:, 2:], dB[:, 2:] = kernels.transfer_kernels(kernels.load_transfer(), x, np.arange(2, 21))
    for name, model in (("sw", kernels.sw_model(20)), ("camb", kernels.radial_model(B, dB, x))):
        _, F = fisher_direct(trispectrum.grid(model, 20), CL)
        assert relative(F, REF[f"F_direct_{name}"]) < 1e-12


def test_fisher_factorisable_sw():
    L, F_direct = fisher_direct(trispectrum.grid(kernels.sw_model(12), 12), CL)
    F = fisher_factorisable(kernels.sw_model(12), CL, L)
    assert relative(F, F_direct) < 1e-10 and relative(F, REF["F_fact_sw"][:11]) < 1e-10
    F_parity = fisher_factorisable(kernels.sw_model(12), CL, [12], parity_weight=True)
    assert abs(0.5 * (F[-1] - F_parity[0]) / F[-1] - 1) < 1e-10        # F_odd = F_unprojected
    if FULL:
        assert relative(fisher_factorisable(kernels.sw_model(20), CL, L.tolist() + list(range(13, 21))),
                        REF["F_fact_sw"]) < 1e-10
        # Cross-machine round-off is ~1e-9 here: the contraction cancels by ~1e5.
        assert abs(fisher_factorisable(kernels.sw_model(100), CL, [100])[0] / REF["F_fact_sw_100"] - 1) < 1e-8


def test_fisher_factorisable_transfer():
    """Factorisable = direct for the full transfer, boundary terms included.

    The identity needs the 3D surface term at x_max to vanish. That term is a
    determinant of B^p_l(x_max) over four distinct l, so it first appears at
    l_max = 6 and decays as x_max grows; here x_max = 50000 Mpc.
    """
    table, chi = kernels.load_transfer(), config.chi_star()
    x = kernels.radial_grid("0:chi:1000,chi:eta0:40,eta0:50000:1000", chi, table["eta0"])
    B, dB = np.zeros((4, 9, len(x))), np.zeros((4, 9, len(x)))
    B[:, 2:], dB[:, 2:] = kernels.transfer_kernels(table, x, np.arange(2, 9))
    model = kernels.radial_model(B, dB, x)
    L, F_direct = fisher_direct(trispectrum.grid(model, 8), CL)
    F = fisher_factorisable(model, CL, L)
    assert relative(F, F_direct) < 1e-10
    split = sum(fisher_factorisable(model, CL, L, nodes=n) for n in np.array_split(np.arange(17), 3))
    assert relative(split, F) < 1e-13                                  # mu chunks add up


class FakeCAMB:
    """Stands in for CAMBdata with a smooth source S(k, eta)."""
    def conformal_time(self, z):
        return 300.0

    def get_time_evolution(self, k, eta, vars, lAccuracyBoost):
        return (np.exp(-((eta - 280.0) / 10.0) ** 2)[None, :] * np.cos(k[:, None] * eta))[:, :, None]


def test_transfer_tasks(tmp_path, monkeypatch):
    for name, value in (("K_BLOCK", 50), ("DELTA_ETA", 2.0), ("K_ETA_MAX", 50.0), ("K_EPSILON", 0.05),
                        ("KAPPA0", 1e-3)):
        monkeypatch.setattr(config, "TRANSFER_" + name, value)
    fake, ells = FakeCAMB(), np.arange(2, 9)
    k, eta = camb_inputs.k_grid(300.0), camb_inputs.eta_grid(300.0)
    whole = camb_inputs.project(fake.get_time_evolution(k, eta, None, 0)[:, :, 0], k, eta, 300.0, ells)
    n_tasks = len(camb_inputs.tasks(len(k), ells, 3))
    for task in range(n_tasks):
        camb_inputs.run_task(fake, task, ells, 3, out_dir=tmp_path)
    camb_inputs.merge(ells, 3, in_dir=tmp_path, out=tmp_path / "transfer.npz")
    with pytest.raises(FileExistsError):
        camb_inputs.merge(ells, 3, in_dir=tmp_path, out=tmp_path / "transfer.npz")
    with np.load(tmp_path / "transfer.npz") as f:
        assert n_tasks > 3 and np.array_equal(f["k_arr"], k)
        np.testing.assert_allclose(f["Tgamma_lk"], whole, rtol=0, atol=1e-15 * np.abs(whole).max())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
