"""Parameters and conventions for the parity-odd CMB trispectrum.

Units: distances in Mpc, wavenumbers in Mpc^-1, spectra in (Delta T / T)^2.
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
CL_TT_FILE = DATA / "Cl_TT.npy"                    # lensed C_ell^TT, starts at ell = 2
CHI_STAR_FILE = DATA / "chi_star.npy"
TRANSFER_FILE = DATA / "Tgamma_lk_camb_source.npz"  # Delta_ell(k) from CAMB T_source

# Planck 2018 cosmology
H0, OMBH2, OMCH2, TAU = 67.32117, 0.0223828, 0.1201075, 0.05430842
A_S, N_S, K_PIVOT = 2.1e-9, 0.9660499, 0.05

# Template powers p_a = (alpha, beta, gamma) - 4 + n_s, plus the physical p = 0 leg
ALPHA, BETA, GAMMA = -2.0, -1.0, 0.0
POWERS = (ALPHA - 4 + N_S, BETA - 4 + N_S, GAMMA - 4 + N_S, 0.0)
COUPLING_G = 2e7

# Instrument: N_ell = (Delta_T / T_CMB)^2 exp(ell^2 / ell_b^2), ell_b = sqrt(8 ln 2) / theta_FWHM
T_CMB_UK = 2.7255e6
NOISE_UK_ARCMIN = 5.0
BEAM_FWHM_ARCMIN = 1.5
F_SKY = 0.4
LMIN = 2

# Radial grid as "start:stop:step" segments in Mpc. "chi" and "eta0" stand for
# the recombination distance and the conformal horizon. Segment ends are nodes.
SMITH_GRID = "0:chi:50,chi:eta0:5,eta0:16000:50"           # Smith et al. (2015) App. D
RADIAL_GRID = "0:chi-500:50,chi-500:chi:5,chi:eta0:5,eta0:16000:50"

# CAMB T_source line-of-sight transfer (Smith et al. 2015, App. D):
# Delta k = min(epsilon k, kappa0); k limits quoted as k * eta0; uniform eta grid.
TRANSFER_K_ETA_MIN = 0.01
TRANSFER_K_ETA_MAX = 5000.0
TRANSFER_KAPPA0 = 3.0e-5
TRANSFER_K_EPSILON = 2.0e-3
TRANSFER_DELTA_ETA = 0.5
TRANSFER_K_BLOCK = 1024        # k values per CAMB source request (sets memory use)
TRANSFER_L_ACCURACY_BOOST = 4


def noise_tt(lmax):
    """White noise with a Gaussian beam, indexed from ell = 0."""
    arcmin = np.pi / (180.0 * 60.0)
    ell_b = np.sqrt(8.0 * np.log(2.0)) / (BEAM_FWHM_ARCMIN * arcmin)
    ell = np.arange(lmax + 1)
    return (NOISE_UK_ARCMIN / T_CMB_UK * arcmin) ** 2 * np.exp(ell**2 / ell_b**2)


def total_spectrum(lmax):
    """C_ell + N_ell indexed from ell = 0; zero below LMIN."""
    cl = np.zeros(lmax + 1)
    cl[LMIN:] = np.load(CL_TT_FILE)[: lmax + 1 - LMIN]
    out = cl + noise_tt(lmax)
    out[:LMIN] = 0.0
    return out


def chi_star():
    return float(np.load(CHI_STAR_FILE)[0])


def zeta_amplitude():
    """Real amplitude A in zeta = (i/2) i^(sum ell) A x (ell-dependent factors)."""
    return (COUPLING_G * (2 * np.pi**2 * A_S) ** 3 * (4 * np.pi) ** 4
            * K_PIVOT ** (3 * (1 - N_S)))


def fisher_prefactor():
    """|i A / 2|^2, the rotation volume 8 pi^2 of int dx dx', and the Wick factor 1/24^2."""
    return 0.25 * zeta_amplitude() ** 2 * 8 * np.pi**2 / 24**2
