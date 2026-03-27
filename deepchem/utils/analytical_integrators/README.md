# analytical_integrators — Pure Python integral implementations

Replaces C/libcint integral functions with pure Python/NumPy/SciPy implementations.

## Files

| File | Purpose |
|------|---------|
| `spherical.py` | Rys quadrature root finding |
| `optimizer.py` | Integral environment setup, optimization, constants |
| `integrals.py` | Integral kernels, drivers, matrix assembly, grid evaluators |

## Dependency Tree

### `spherical.py` — Rys quadrature roots

```
rys_roots(nroots, x)
├── _boys(mmax, x)
├── _poly(a, x)
├── _schmidt(f, n)
└── _find_roots(coeffs, rt, tol)
    └── _poly(a, x)
```

### `optimizer.py` — Integral environment setup & optimization

```
CINTcommon_fac_sp(l)                          # leaf — normalization factor

CINTcart_comp(lmax)                           # leaf — Cartesian component indices

CINTEnvVars                                   # class — integral environment container
PairData                                      # class — primitive pair data
CINTOpt                                       # class — optimizer container

approx_log(x)                                 # leaf

CINTinit_int1e_EnvVars(envs, ng, shls, ...)   # leaf — 1e environment init
CINTinit_int2e_EnvVars(envs, ng, shls, ...)   # 2e environment init
└── CINTcommon_fac_sp
CINTinit_int3c2e_EnvVars(envs, ng, shls, ...) # 3c2e environment init
└── CINTcommon_fac_sp

CINTg1e_index_xyz(envs)                       # 1e index mapping
└── CINTcart_comp
CINTg2e_index_xyz(envs)                       # 2e index mapping
└── CINTcart_comp

CINTset_pairdata(pairdata, ai, aj, ...)
└── approx_log

_numpy_vec_log_maxc(log_maxc, coeff, ...)     # leaf

CINTOpt_set_log_maxc(opt, atm, ...)
└── _numpy_vec_log_maxc
CINTOpt_non0coeff_byshell(ci, iprim, ...)     # leaf
CINTOpt_set_non0coeff(opt, atm, ...)
└── CINTOpt_non0coeff_byshell
CINTOpt_setij(opt, ng, atm, ...)
├── CINTOpt_set_log_maxc
├── CINTset_pairdata
│   └── approx_log
└── PairData

gen_idx(opt, ng, atm, ...)
├── CINTEnvVars
└── _make_fakebas                             # leaf

CINTall_1e_optimizer(opt, ng, atm, ...)
├── CINTOpt_set_log_maxc
│   └── _numpy_vec_log_maxc
├── CINTOpt_set_non0coeff
│   └── CINTOpt_non0coeff_byshell
└── gen_idx

CINTall_2e_optimizer(opt, ng, atm, ...)
├── CINTOpt_set_non0coeff
├── CINTOpt_setij
└── gen_idx

CINTinit_2e_optimizer(atm, ...)               # creates CINTOpt
└── CINTOpt

── 1e Optimizers ──────────────────────────
int1e_ovlp_optimizer(opt_ref, atm, ...)
├── CINTinit_2e_optimizer
└── CINTall_1e_optimizer

int1e_kin_optimizer(opt_ref, atm, ...)
├── CINTinit_2e_optimizer
└── CINTall_1e_optimizer

int1e_nuc_optimizer(opt_ref, atm, ...)
├── CINTinit_2e_optimizer
└── CINTall_1e_optimizer

── 2e/3c Optimizers ───────────────────────
int2e_ar12b_optimizer(opt_ref, atm, ...)
├── CINTinit_2e_optimizer
└── CINTall_2e_optimizer

int3c2e_ar12_optimizer(opt_ref, atm, ...)
├── CINTinit_2e_optimizer
├── CINTOpt_set_non0coeff
├── CINTOpt_setij
└── gen_idx
```

### `integrals.py` — Integral kernels, drivers & grid evaluators

```
── Cart-to-Spherical ──────────────────────
_cart2sph_matrix(l)                           # leaf
cart2sph_matrix(l)                            # cached wrapper
└── _cart2sph_matrix

c2s_sph_1e(gctr, i_l, j_l, ...)
└── cart2sph_matrix
c2s_sph_2e1(gctr, i_l, j_l, k_l, l_l, ...)
└── cart2sph_matrix
c2s_sph_3c2e1(gctr, i_l, j_l, k_l, ...)
└── cart2sph_matrix

── 1e Integral Primitives ─────────────────
CINTg_ovlp(g, ai, aj, fac, envs)             # leaf — overlap g-values
CINTg_nuc(g, aij, rij, cr, t2, fac, envs)    # leaf — nuclear g-values
CINTnabla1j_1e(f, g, li, lj, lk, envs)       # leaf — nabla operator

── 1e Gout Functions ──────────────────────
gout_1e_ovlp(gout, g, idx, envs)              # leaf
gout_1e_nuc(gout, g, idx, envs)
└── gout_1e_ovlp
gout_1e_kin(gout, g, idx, envs)
└── CINTnabla1j_1e

── Primitive-to-Contracted ────────────────
CINTprim_to_ctr(gc, nf, gp, ...)             # leaf
CINTprim_to_ctr_simple(gc, nf, gp, ...)      # leaf

── 1e Integral Loops & Drivers ────────────
CINT1e_loop(envs, atm, bas, env)              # overlap/kinetic loop
├── CINTcommon_fac_sp          [from optimizer]
├── CINTg1e_index_xyz          [from optimizer]
└── CINTg_ovlp

CINT1e_nuc_loop(envs, atm, bas, env, ...)     # nuclear attraction loop
├── CINTcommon_fac_sp          [from optimizer]
├── CINTg1e_index_xyz          [from optimizer]
├── CINTg_nuc
└── rys_roots                  [from spherical]

CINT1e_drv(envs, atm, bas, env, int1e_type)   # 1e driver
├── CINT1e_loop
├── CINT1e_nuc_loop
└── c2s_sph_1e

── 1e Entry Points ────────────────────────
int1e_ovlp_sph(out, dims, shls, ...)          # OVERLAP
├── CINTEnvVars                [from optimizer]
├── CINTinit_int1e_EnvVars     [from optimizer]
└── CINT1e_drv

int1e_kin_sph(out, dims, shls, ...)           # KINETIC ENERGY
├── CINTEnvVars
├── CINTinit_int1e_EnvVars
└── CINT1e_drv

int1e_nuc_sph(out, dims, shls, ...)           # NUCLEAR ATTRACTION
├── CINTEnvVars
├── CINTinit_int1e_EnvVars
└── CINT1e_drv

── 2e Integral Primitives ─────────────────
CINTg0_2e_2d(g, bc, envs)                    # leaf — 2D recurrence
CINTg0_lj2d_4d(g, envs)                      # leaf — l,j 2D->4D
CINTg0_kj2d_4d(g, envs)                      # leaf — k,j 2D->4D
CINTg0_ik2d_4d(g, envs)                      # leaf — i,k 2D->4D
CINTg0_il2d_4d(g, envs)                      # leaf — i,l 2D->4D

CINTg0_2e(g, fac, envs)                      # full 2e g-tensor
├── CINTg0_2e_2d
├── CINTg0_lj2d_4d
├── CINTg0_kj2d_4d
├── CINTg0_ik2d_4d
├── CINTg0_il2d_4d
└── rys_roots                  [from spherical]

── 2e Gout Function ───────────────────────
gout_2e_ar12b(gout, g, idx, envs, ...)        # leaf

── 2e Integral Loops & Drivers ────────────
CINT2e_loop_nopt(envs, atm, bas, env)
├── CINTg0_2e
├── CINTg2e_index_xyz          [from optimizer]
└── gout_2e_ar12b

CINT2e_spheric_drv(envs, atm, bas, env)
├── CINT2e_loop_nopt
└── c2s_sph_2e1

── 2e Entry Point ─────────────────────────
int2e_ar12b_sph(out, dims, shls, ...)         # 4c ERI
├── CINTEnvVars
├── CINTinit_int2e_EnvVars     [from optimizer]
└── CINT2e_spheric_drv

── 2e Matrix Assembly ─────────────────────
GTOint2c(intor, out, ...)                     # leaf — 2-center driver
GTOnr2e_fill_s1(intor, eri, ...)              # leaf — 4-center fill
GTOnr2e_fill_drv(intor, fill, eri, ...)       # leaf — 4-center driver

── 3c2e Integral Loops & Drivers ──────────
CINT3c2e_loop_nopt(envs, atm, bas, env)
├── CINTg0_2e
├── CINTg2e_index_xyz          [from optimizer]
└── gout_2e_ar12b

CINT3c2e_spheric_drv(envs, atm, bas, env)
├── CINT3c2e_loop_nopt
└── c2s_sph_3c2e1

── 3c2e Entry Point ───────────────────────
int3c2e_ar12_sph(out, dims, shls, ...)        # 3-center 2e integral
├── CINTEnvVars
├── CINTinit_int3c2e_EnvVars   [from optimizer]
└── CINT3c2e_spheric_drv

── 3c2e Matrix Assembly ───────────────────
GTOnr3c_fill_s1(intor, out, ...)              # leaf — 3-center fill
GTOnr3c_drv(intor, fill, eri, ...)            # leaf — 3-center driver

── Fourier Transform GTO Evaluator ────────
_ft_1d_poly(k, n, a2)                        # leaf — 1D FT polynomial

gto_ft_evaluator_py(wrapper, gvgrid)          # FT of GTO basis
├── CINTcart_comp              [from optimizer]
├── CINTcommon_fac_sp          [from optimizer]
├── _ft_1d_poly
└── cart2sph_matrix

── Real-Space GTO Grid Evaluator ──────────
gto_evaluator_py_grid(wrapper, shortname, rgrid, spherical)
├── CINTcommon_fac_sp          [from optimizer]
└── cart2sph_matrix
```

## Cross-file imports

```
integrals.py ──imports from──> optimizer.py
  CINTEnvVars, CINTcart_comp, CINTcommon_fac_sp,
  CINTinit_int1e_EnvVars, CINTg1e_index_xyz,
  CINTinit_int2e_EnvVars, CINTg2e_index_xyz,
  CINTinit_int3c2e_EnvVars,
  ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF,
  PTR_COORD, EXPCUTOFF, MIN_EXPCUTOFF, PTR_EXPCUTOFF,
  BAS_SLOTS, ATM_SLOTS

integrals.py ──imports from──> spherical.py
  rys_roots
```

## External callers

These functions are called from `deepchem/utils/dft_utils/hamilton/intor/`:

| Caller file | Functions used |
|-------------|---------------|
| `molintor.py` | `int3c2e_ar12_optimizer`, `GTOnr3c_drv`, `GTOnr3c_fill_s1` (from integrals); optimizer functions for 1e/2e/3c2e |
| `gtoft.py` | `gto_ft_evaluator_py` (from integrals) |
| `gtoeval.py` | `gto_evaluator_py_grid` (from integrals) |
| `lcintwrap.py` | Integral entry points via `INTEGRAL_REGISTRY`, optimizer functions |
