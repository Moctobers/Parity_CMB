# Parity-odd CMB trispectrum: reduced trispectrum and SNR

Computes the reduced trispectrum of the parity-odd contact template, in the
Sachs-Wolfe limit and with the full CMB transfer function, and its Fisher
information (= SNR² for one amplitude) by two methods: the direct sum over a
complete grid of ζ, and the factorisable Fisher. Equations follow
`notes/parity-cmb/main.tex`; each module docstring names the equations it uses.

Needs only numpy and scipy (camb for `camb_inputs.py`).

| File | Contents |
|---|---|
| `config.py` | cosmology, template, noise, radial grid, transfer settings, prefactors |
| `terms.py` | the 216 bulk + 36 boundary signed terms after IBP |
| `kernels.py` | B, ∂ₓB (SW closed form or k-integrals of Δ_ℓ(k)), radial grid, radial model |
| `trispectrum.py` | 3j recursion, ζ for one configuration, complete grid |
| `fisher_direct.py` | F(L) = (1/24) Σ \|ζ\|² / ∏(C+N) |
| `fisher_factorisable.py` | Wigner-d recursion, Z, permanent contraction, one-pass F(L) |
| `run.py` | command line: kernels, direct, fisher (with array-job chunks) |
| `camb_inputs.py` | CAMB C_ℓ^TT, χ*, and Δ_ℓ(k) from T_source (with array-job tasks) |
| `plot.py` | figures: kernels over the noise, the ζ ladder, SNR curves |
| `test_core.py` | checks; `tests/reference.npz` holds old-code outputs |

## Conventions

- Kernels are phase-stripped: B̂ = (1/2π²) ∫dk k^(p+2) j_ℓ(kx) Δ_ℓ(k); i^Σℓ is in the ζ prefactor.
- SW: the p = 0 leg is a shell, so the radial model has one node at χ* with weight
  −1/(20πχ*²) in the p = 0 row and no boundary terms. Full transfer: trapezoid
  bulk plus the IBP boundary terms at x_min (−1) and x_max (+1).
- ζ is stored for odd Σℓ only (it vanishes for even Σℓ), all orderings of ℓ₁..ℓ₄,
  and the full triangle range of the internal ℓ*.
- F is full sky. `run.py` stores SNR = √(f_sky F) with `--f-sky` (default 0.4).
- The factorisable Fisher uses only the unprojected Z, since F_odd = F_unprojected.

## Running

```sh
python run.py fisher --source sw --lmax 100 --ell-step 10
python run.py kernels --lmax 20                 # full-transfer kernels on config.RADIAL_GRID
python run.py direct --source transfer --lmax 20
python run.py fisher --source transfer --lmax 20
```

Figures (written to `output/`):

```sh
python plot.py kernels --lmax 20 --sources sw transfer       # B² and (∂ₓB)² over C+N at χ*
python plot.py zeta --lmax 20 --sources sw transfer [--index-range 0 300]
python plot.py snr --lmax 20 --curves direct:sw direct:transfer fisher:sw fisher:transfer
```

On a cluster, the expensive steps split into independent tasks with
`--n-chunks N --task $SLURM_ARRAY_TASK_ID`, followed by one `--merge` run with the
same arguments: `kernels` splits over ℓ, `fisher` over Gauss-Legendre nodes, and
`camb_inputs.py transfer` over (k block × ℓ chunk) with `--n-l-chunks`.

## Transfer table on the cluster (ℓ ≤ 2000)

`cluster/` holds the SLURM scripts. On the login node, with the camb environment active:

```sh
cluster/submit_transfer.sh test      # one core, ~15 min: timing, memory, check vs the stored table, batch plan
cluster/submit_transfer.sh run       # array job, one task per k block (1 core), then a merge job
cluster/submit_transfer.sh missing   # resubmit the tasks that left no file, then merge again
```

Tasks split k, not ℓ: every ℓ chunk re-runs CAMB for its k block. The projection cost per k
grows with k (~6 s for all ℓ ≤ 2000 at kη₀ < 5, ~330 s at kη₀ = 5000, on a 2.1 GHz Xeon core),
so the highest-k tasks are submitted first. CAMB fills all 24 evolve variables (5.4 MB per k),
which sets the memory per task (`config.TRANSFER_K_BLOCK`). The table goes to
`data/Tgamma_lk_camb_source_lmax2000.npz`; point `config.TRANSFER_FILE` at it to use it.

## Tests

```sh
python -m pytest -q test_core.py            # ~2 min
FULL=1 python -m pytest -q test_core.py     # adds the ℓ ≤ 20 references
```

## Open questions

- Radial grid: x_max and the sampling beyond η₀ change the CAMB SNR(20) by ~0.3%
  (SNR(8) by ~2%). Beyond η₀, B⁰ is only k-cutoff ringing (period ~18 Mpc).
  Direct and factorisable F agree only when the 3D surface term at x_max is
  negligible (it needs four distinct ℓ, so it first appears at ℓ_max = 6).
- The stored transfer covers ℓ = 2..50 and k ≤ 0.353 Mpc⁻¹; k_max must be checked
  before an ℓ ≈ 2000 table is made.
- `camb_inputs.py transfer` uses `camb.get_background` per task; check it on a few
  rows against the existing table before a production run.
