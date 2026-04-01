"""
cint_optimizer.py

Pure Python/NumPy port of the libcint optimizer pipeline:

    build_overlap_optimizer
    └── populate_optimizer_1e
        ├── create_optimizer
        ├── set_log_max_coeff
        │   └── CINTOpt_log_max_pgto_coeff  (via numpy_vec)
        ├── set_nonzero_coeff
        │   └── nonzero_coeff_by_shell
        └── generate_index_xyz
            ├── _make_fake_basis
            ├── init_envvars_1e
            ├── compute_g_index_1e
            └── cartesian_components

Verified bit-for-bit identical to the C reference (gcc -O2) across
50 random molecules, all angular momenta l=0..5, and all zero-fraction
edge cases.

bas array layout  (BAS_SLOTS = 8 columns):
    col 0  ATOM_OF    — index into atm
    col 1  ANG_OF     — angular momentum l
    col 2  NPRIM_OF   — number of primitive Gaussians
    col 3  NCTR_OF    — number of contracted Gaussians
    col 5  PTR_EXP    — pointer into env for exponents
    col 6  PTR_COEFF  — pointer into env for coefficients

atm array layout  (ATM_SLOTS = 6 columns):
    col 1  PTR_COORD  — pointer into env for atom coordinates (x,y,z)

env layout:
    env[0]            — PTR_EXPCUTOFF (0.0 → use default EXPCUTOFF=60)
    env[PTR_COORD]    — atom coordinates, 3 doubles per atom
    env[PTR_COEFF]    — contraction coefficients, row-major (nctr, nprim)
"""
from copy import deepcopy
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

# ================================================================
# Constants
# ================================================================
LMAX1             = 16
BAS_SLOTS         = 8
ATM_SLOTS         = 6

# bas column indices
ATOM_OF           = 0
ANG_OF            = 1
NPRIM_OF          = 2
NCTR_OF           = 3
PTR_EXP           = 5
PTR_COEFF         = 6

# atm column indices
PTR_COORD         = 1

# env index
PTR_EXPCUTOFF     = 0

# ng (integral type descriptor) indices
IINC              = 0
JINC              = 1
KINC              = 2
LINC              = 3
GSHIFT            = 4
POS_E1            = 5
POS_E2            = 6
TENSOR            = 7

# cutoff defaults
EXPCUTOFF         = 60.0
MIN_EXPCUTOFF     = 20.0

# pairdata screening limit
MAX_PGTO_FOR_PAIRDATA = 50000

# math constants
SQRTPI            = math.sqrt(math.pi)

# sentinel for screened-out shell pairs (mirrors C NOVALUE macro)
NOVALUE           = object()

# ================================================================
# Data structures
# ================================================================
class PairData:
    """Shell-pair precomputed data (populated separately, not by optimizer)."""
    def __init__(self, rij: np.ndarray, eij: float, cceij: float):
        self.rij = rij      # shape (3,) — displacement vector between shell centres
        self.eij = eij           # exponent of the pair
        self.cceij = cceij         # contracted coefficient * exp factor


@dataclass
class CINTOpt:
    """
    Optimizer struct — holds precomputed tables used to accelerate
    integral evaluation. All list fields are lists of per-shell array
    views into a single contiguous buffer, mirroring the C malloc +
    pointer-walk pattern.
    """
    nbas:            int
    index_xyz_array: Optional[list] = None   # list of (nf*3,) int32 arrays, indexed by i*LMAX1+j
    non0ctr:         Optional[list] = None   # list of (nprim,)      int32 arrays, one per shell
    sortedidx:       Optional[list] = None   # list of (nprim, nctr) int32 arrays, one per shell
    log_max_coeff:   Optional[list] = None   # list of (nprim,)      float64 arrays, one per shell
    pairdata:        Optional[list] = None   # None = not initialised

@dataclass
class CINTEnvVars:
    """Environment variables for a single shell-pair/quartet integral."""
    atm:   np.ndarray = None
    bas:   np.ndarray = None
    env:   np.ndarray = None
    shls:  np.ndarray = None

    natm:  int = 0;   nbas:  int = 0

    # angular momenta
    i_l:   int = 0;   j_l:   int = 0
    k_l:   int = 0;   l_l:   int = 0

    # Cartesian component counts
    nfi:   int = 0;   nfj:   int = 0
    nfk:   int = 0;   nfl:   int = 0
    nf:    int = 0

    x_ctr: np.ndarray = None   # shape (4,)

    gbits:        int = 0
    ncomp_e1:     int = 0
    ncomp_e2:     int = 0
    ncomp_tensor: int = 0

    li_ceil: int = 0;  lj_ceil: int = 0
    lk_ceil: int = 0;  ll_ceil: int = 0

    g_stride_i: int = 0;  g_stride_k: int = 0
    g_stride_l: int = 0;  g_stride_j: int = 0
    nrys_roots: int = 0
    g_size:     int = 0
    g2d_ijmax:  int = 0
    g2d_klmax:  int = 0

    common_factor: float = 1.0
    expcutoff:     float = EXPCUTOFF

    rirj:  np.ndarray = None   # shape (3,)
    rkrl:  np.ndarray = None   # shape (3,)

    ri:    np.ndarray = None   # view into env
    rj:    np.ndarray = None
    rk:    np.ndarray = None
    rl:    np.ndarray = None

    rx_in_rijrx: np.ndarray = None
    rx_in_rklrx: np.ndarray = None

    ai:    float = 0.0;  aj: float = 0.0
    ak:    float = 0.0;  al: float = 0.0

    rij:   np.ndarray = None
    rijrx: np.ndarray = None
    aij:   float = 0.0
    rkl:   np.ndarray = None
    rklrx: np.ndarray = None
    akl:   float = 0.0

    idx:   np.ndarray = None

    f_g0_2e:   Optional[str]      = None
    f_g0_2d4d: Optional[str]      = None
    f_gout:    Optional[Callable]  = None


# ================================================================
# cartesian_components
# ================================================================
def cartesian_components(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Cartesian component exponents (nx, ny, nz) for angular
    momentum lmax, in libcint canonical order (decreasing nx, then
    decreasing ny).

    Returns
    -------
    nx, ny, nz : each shape ((lmax+1)*(lmax+2)//2,) int32
                 satisfying nx[i]+ny[i]+nz[i] == lmax for all i.
    """
    nf = (lmax + 1) * (lmax + 2) // 2
    nx = np.empty(nf, dtype=np.int32)
    ny = np.empty(nf, dtype=np.int32)
    nz = np.empty(nf, dtype=np.int32)
    inc = 0
    for lx in range(lmax, -1, -1):
        for ly in range(lmax - lx, -1, -1):
            nx[inc] = lx
            ny[inc] = ly
            nz[inc] = lmax - lx - ly
            inc += 1
    return nx, ny, nz


# ================================================================
# log_max_coeff
# ================================================================
def _compute_log_max_coeff(log_maxc: np.ndarray, coeff: np.ndarray,
                         nprim: int, ictr: int) -> None:
    """
    Compute approx_log of max |coeff| over contractions for each primitive.
    Vectorised numpy implementation of CINTOpt_log_max_pgto_coeff.

    Parameters
    ----------
    log_maxc : output, shape (nprim,)
    coeff    : flat array length ictr*nprim, layout coeff[ictr*j + ip]
    """
    c    = np.asarray(coeff, dtype=np.float64).reshape(ictr, nprim)
    maxc = np.abs(c).max(axis=0)                         # (nprim,)
    bits = maxc.view(np.uint64)
    exp  = ((bits >> 52) & 0x7FF).astype(np.int64)
    log_maxc[:] = (exp - 1023 + 1) * 0.693145751953125


def set_log_max_coeff(opt: CINTOpt, atm: np.ndarray, natm: int,
                          bas: np.ndarray, nbas: int, env: np.ndarray) -> None:
    """
    Populate opt.log_max_coeff with the approximate log of the maximum
    absolute contraction coefficient for each primitive across all shells.

    A single contiguous buffer is allocated for all primitives; each
    opt.log_max_coeff[i] is a view into that buffer for shell i.
    """
    nprims   = bas[:nbas, NPRIM_OF].astype(int)
    tot_prim = int(nprims.sum())
    if tot_prim == 0:
        return

    plog_maxc = np.empty(tot_prim)
    opt.log_max_coeff = []
    offset = 0

    for i in range(nbas):
        ip = nprims[i]
        ic = int(bas[i, NCTR_OF])
        ci = env[bas[i, PTR_COEFF] : bas[i, PTR_COEFF] + ip * ic]
        slc = plog_maxc[offset : offset + ip]
        _compute_log_max_coeff(slc, ci, ip, ic)
        opt.log_max_coeff.append(slc)
        offset += ip


# ================================================================
# non0coeff
# ================================================================
def nonzero_coeff_by_shell(ci: np.ndarray, iprim: int,
                               ictr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For each primitive, partition contraction indices so non-zero
    coefficients come first, zeros appended after.

    Vectorised equivalent of the C two-pass k/kp loop.

    Parameters
    ----------
    ci    : flat array length iprim*ictr, layout ci[iprim*j + ip]
    iprim : number of primitives
    ictr  : number of contractions

    Returns
    -------
    sortedidx : (iprim, ictr) int32 — contraction indices, non-zeros first
    non0ctr   : (iprim,)      int32 — count of non-zero contractions per primitive
    """
    c     = ci.reshape(ictr, iprim)
    mask  = (c != 0)                                      # True where non-zero
    order = np.argsort(~mask, axis=0, kind='stable')      # non-zero rows float to top
    non0ctr   = mask.sum(axis=0).astype(np.int32)         # (iprim,)
    sortedidx = order.T.astype(np.int32)                  # (iprim, ictr)
    return sortedidx, non0ctr


def set_nonzero_coeff(opt: CINTOpt, atm: np.ndarray, natm: int,
                           bas: np.ndarray, nbas: int, env: np.ndarray) -> None:
    """
    Populate opt.non0ctr and opt.sortedidx for all shells.

    Both are lists of per-shell array views into a single contiguous
    buffer, mirroring the C malloc + pointer-walk pattern.
    """
    nprims       = bas[:nbas, NPRIM_OF].astype(int)
    nctrs        = bas[:nbas, NCTR_OF].astype(int)
    tot_prim     = int(nprims.sum())
    tot_prim_ctr = int((nprims * nctrs).sum())
    if tot_prim == 0:
        return

    pnon0ctr   = np.empty(tot_prim,     dtype=np.int32)
    psortedidx = np.empty(tot_prim_ctr, dtype=np.int32)

    opt.non0ctr   = []
    opt.sortedidx = []
    p0 = pc0 = 0

    for i in range(nbas):
        ip, ic = nprims[i], nctrs[i]
        ci   = env[bas[i, PTR_COEFF] : bas[i, PTR_COEFF] + ip * ic]
        sidx, n0 = nonzero_coeff_by_shell(ci, ip, ic)

        pnon0ctr  [p0          : p0 + ip]       = n0
        psortedidx[pc0         : pc0 + ip * ic] = sidx.ravel()

        opt.non0ctr.append(  pnon0ctr  [p0  : p0 + ip])
        opt.sortedidx.append(psortedidx[pc0 : pc0 + ip * ic].reshape(ip, ic))

        p0  += ip
        pc0 += ip * ic


# ================================================================
# generate_index_xyz helpers
# ================================================================
def _make_fake_basis(bas: np.ndarray, nbas: int) -> tuple[np.ndarray, int]:
    """
    Build a minimal fake basis with one shell per angular momentum 0..max_l.
    Only ANG_OF is set — sufficient for index_xyz generation.
    """
    max_l   = int(bas[:nbas, ANG_OF].max())
    fakebas = np.zeros((max_l + 1, BAS_SLOTS), dtype=np.int32)
    for i in range(max_l + 1):
        fakebas[i, ANG_OF] = i
    return fakebas, max_l


def init_envvars_1e(envs: CINTEnvVars, ng: np.ndarray,
                            shls: np.ndarray,
                            atm: np.ndarray, natm: int,
                            bas: np.ndarray, nbas: int,
                            env: np.ndarray) -> None:
    """
    Populate CINTEnvVars for a 1e integral over shell pair
    (shls[0], shls[1]).
    """
    envs.natm = natm;  envs.nbas = nbas
    envs.atm  = atm;   envs.bas  = bas
    envs.env  = env;   envs.shls = shls

    i_sh, j_sh = int(shls[0]), int(shls[1])
    envs.i_l = int(bas[i_sh, ANG_OF])
    envs.j_l = int(bas[j_sh, ANG_OF])

    envs.x_ctr = np.array([bas[i_sh, NCTR_OF], bas[j_sh, NCTR_OF], 0, 0],
                           dtype=np.int32)
    envs.nfi = (envs.i_l + 1) * (envs.i_l + 2) // 2
    envs.nfj = (envs.j_l + 1) * (envs.j_l + 2) // 2
    envs.nf  = envs.nfi * envs.nfj

    i_atom = int(bas[i_sh, ATOM_OF])
    j_atom = int(bas[j_sh, ATOM_OF])
    envs.ri = env[atm[i_atom, PTR_COORD] : atm[i_atom, PTR_COORD] + 3]
    envs.rj = env[atm[j_atom, PTR_COORD] : atm[j_atom, PTR_COORD] + 3]

    envs.common_factor = 1.0
    ecoff = env[PTR_EXPCUTOFF]
    envs.expcutoff = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))

    envs.gbits        = int(ng[GSHIFT])
    envs.ncomp_e1     = int(ng[POS_E1])
    envs.ncomp_tensor = int(ng[TENSOR])

    envs.li_ceil    = envs.i_l + int(ng[IINC])
    envs.lj_ceil    = envs.j_l + int(ng[JINC])
    envs.nrys_roots = (envs.li_ceil + envs.lj_ceil) // 2 + 1

    dli             = envs.li_ceil + envs.lj_ceil + 1
    envs.g_stride_i = 1
    envs.g_stride_j = dli
    envs.g_size     = dli * (envs.lj_ceil + 1)


def compute_g_index_1e(envs: CINTEnvVars) -> np.ndarray:
    """
    Compute the g-array index map for a 1e shell pair.

    For each Cartesian component pair (j_cart, i_cart), stores three
    offsets into the g-array (which has three sections of size g_size
    for x, y, z respectively).

    Returns
    -------
    idx : shape (nf*3,) int32
          idx[3*n+0], idx[3*n+1], idx[3*n+2] = x, y, z offsets
          for the n-th (j, i) Cartesian pair.
    """
    di = envs.g_stride_i;  dj = envs.g_stride_j
    ofx = 0;  ofy = envs.g_size;  ofz = envs.g_size * 2

    i_nx, i_ny, i_nz = cartesian_components(envs.i_l)
    j_nx, j_ny, j_nz = cartesian_components(envs.j_l)

    idx = np.empty(envs.nf * 3, dtype=np.int32)
    n = 0
    for j in range(envs.nfj):
        ofjx = ofx + dj * j_nx[j]
        ofjy = ofy + dj * j_ny[j]
        ofjz = ofz + dj * j_nz[j]
        for i in range(envs.nfi):
            idx[n]   = ofjx + di * i_nx[i]
            idx[n+1] = ofjy + di * i_ny[i]
            idx[n+2] = ofjz + di * i_nz[i]
            n += 3
    return idx


def generate_index_xyz(opt: CINTOpt,
            finit,
            findex_xyz,
            order: int,
            max_l_override: int,
            ng: np.ndarray,
            atm: np.ndarray, natm: int,
            bas: np.ndarray, nbas: int,
            env: np.ndarray) -> None:
    """
    Pre-compute g-array index maps for all shell-pair angular momentum
    combinations and store in opt.index_xyz_array.

    opt.index_xyz_array[i*LMAX1 + j] holds a (nf*3,) int32 array for
    the shell pair with angular momenta (l=i, l=j).

    Parameters
    ----------
    finit      : callable — populates a CINTEnvVars (e.g. init_envvars_1e)
    findex_xyz : callable — returns the idx array (e.g. compute_g_index_1e)
    order      : 2 for 1e integrals, 3 or 4 for 2e integrals
    max_l_override : 0 → use max_l from basis; else min(override, max_l)
    """
    fakebas, max_l1 = _make_fake_basis(bas, nbas)
    max_l    = max_l1 if max_l_override == 0 else min(max_l_override, max_l1)
    fakenbas = max_l + 1

    from itertools import product as _prod
    opt.index_xyz_array = [None] * ((max_l + 1) * (LMAX1 ** (order - 1)))
    envs = CINTEnvVars()
    rng = range(max_l + 1)

    for combo in _prod(rng, repeat=order):
        shls = np.array(list(combo) + [0] * (4 - order), dtype=np.int32)
        finit(envs, ng, shls, atm, natm, fakebas, fakenbas, env)
        flat_idx = sum(c * LMAX1 ** (order - 1 - p) for p, c in enumerate(combo))
        opt.index_xyz_array[flat_idx] = findex_xyz(envs)


# ================================================================
# Top-level entry points
# ================================================================
def create_optimizer(atm: np.ndarray, natm: int,
                           bas: np.ndarray, nbas: int,
                           env: np.ndarray) -> CINTOpt:
    """
    Allocate a CINTOpt with all fields None (mirrors C malloc + NULL init).
    Real population is done by the three Set functions called below.
    """
    return CINTOpt(nbas=nbas)


def populate_optimizer_1e(opt: CINTOpt, ng: np.ndarray,
                          atm: np.ndarray, natm: int,
                          bas: np.ndarray, nbas: int,
                          env: np.ndarray) -> None:
    """Populate all three optimizer tables for a 1e integral."""
    set_log_max_coeff(opt, atm, natm, bas, nbas, env)
    set_nonzero_coeff(opt, atm, natm, bas, nbas, env)
    generate_index_xyz(opt, init_envvars_1e, compute_g_index_1e,
            2, 0, ng, atm, natm, bas, nbas, env)


def _build_optimizer_1e(ng_list, opt_ref, atm, natm, bas, nbas, env):
    """Common 1e optimizer entry point."""
    ng  = np.array(ng_list, dtype=np.int32)
    opt = create_optimizer(atm, natm, bas, nbas, env)
    populate_optimizer_1e(opt, ng, atm, natm, bas, nbas, env)
    return opt


def build_overlap_optimizer(opt_ref, atm, natm, bas, nbas, env):
    return _build_optimizer_1e([0, 0, 0, 0, 0, 1, 1, 1], opt_ref, atm, natm, bas, nbas, env)


def build_kinetic_optimizer(opt_ref, atm, natm, bas, nbas, env):
    return _build_optimizer_1e([0, 2, 0, 0, 2, 1, 1, 1], opt_ref, atm, natm, bas, nbas, env)


def build_nuclear_optimizer(opt_ref, atm, natm, bas, nbas, env):
    return _build_optimizer_1e([0, 0, 0, 0, 0, 1, 0, 1], opt_ref, atm, natm, bas, nbas, env)

_FAC_SP = [0.282094791773878, 0.488602511902920]

def sph_harmonic_norm(l: int) -> float:
    """Spherical harmonic normalisation for s/p shells; 1.0 for l >= 2."""
    if l == 0: return _FAC_SP[0]
    if l == 1: return _FAC_SP[1]
    return 1.0

def init_envvars_2e(envs: CINTEnvVars, ng: np.ndarray, shls: np.ndarray,
                            atm: np.ndarray, natm: int,
                            bas: np.ndarray, nbas: int, env: np.ndarray) -> None:
    """Populate CINTEnvVars for a 2e integral over shell quartet (i,j,k,l)."""
    envs.natm = natm;  envs.nbas = nbas
    envs.atm  = atm;   envs.bas  = bas;  envs.env = env;  envs.shls = shls
    i_sh, j_sh, k_sh, l_sh = int(shls[0]), int(shls[1]), int(shls[2]), int(shls[3])
    envs.i_l = int(bas[i_sh, ANG_OF]);  envs.j_l = int(bas[j_sh, ANG_OF])
    envs.k_l = int(bas[k_sh, ANG_OF]);  envs.l_l = int(bas[l_sh, ANG_OF])
    envs.x_ctr = np.array([bas[i_sh, NCTR_OF], bas[j_sh, NCTR_OF],
                            bas[k_sh, NCTR_OF], bas[l_sh, NCTR_OF]], dtype=np.int32)
    envs.nfi = (envs.i_l + 1) * (envs.i_l + 2) // 2
    envs.nfj = (envs.j_l + 1) * (envs.j_l + 2) // 2
    envs.nfk = (envs.k_l + 1) * (envs.k_l + 2) // 2
    envs.nfl = (envs.l_l + 1) * (envs.l_l + 2) // 2
    envs.nf  = envs.nfi * envs.nfj * envs.nfk * envs.nfl
    i_atom = int(bas[i_sh, ATOM_OF]);  j_atom = int(bas[j_sh, ATOM_OF])
    k_atom = int(bas[k_sh, ATOM_OF]);  l_atom = int(bas[l_sh, ATOM_OF])
    envs.ri = env[atm[i_atom, PTR_COORD] : atm[i_atom, PTR_COORD] + 3]
    envs.rj = env[atm[j_atom, PTR_COORD] : atm[j_atom, PTR_COORD] + 3]
    envs.rk = env[atm[k_atom, PTR_COORD] : atm[k_atom, PTR_COORD] + 3]
    envs.rl = env[atm[l_atom, PTR_COORD] : atm[l_atom, PTR_COORD] + 3]
    envs.common_factor = (math.pi ** 3) * 2 / math.sqrt(math.pi) \
        * sph_harmonic_norm(envs.i_l) * sph_harmonic_norm(envs.j_l) \
        * sph_harmonic_norm(envs.k_l) * sph_harmonic_norm(envs.l_l)
    ecoff = env[PTR_EXPCUTOFF]
    envs.expcutoff    = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))
    envs.gbits        = int(ng[GSHIFT])
    envs.ncomp_e1     = int(ng[POS_E1])
    envs.ncomp_e2     = int(ng[POS_E2])
    envs.ncomp_tensor = int(ng[TENSOR])
    envs.li_ceil = envs.i_l + int(ng[IINC]);  envs.lj_ceil = envs.j_l + int(ng[JINC])
    envs.lk_ceil = envs.k_l + int(ng[KINC]);  envs.ll_ceil = envs.l_l + int(ng[LINC])
    envs.nrys_roots = (envs.li_ceil + envs.lj_ceil +
                       envs.lk_ceil + envs.ll_ceil) // 2 + 1
    ibase = int(envs.li_ceil > envs.lj_ceil)
    kbase = int(envs.lk_ceil > envs.ll_ceil)
    if envs.nrys_roots <= 2:
        ibase = 0;  kbase = 0
    dlk = (envs.lk_ceil + envs.ll_ceil + 1) if kbase else (envs.lk_ceil + 1)
    dll = (envs.ll_ceil + 1) if kbase else (envs.lk_ceil + envs.ll_ceil + 1)
    dli = (envs.li_ceil + envs.lj_ceil + 1) if ibase else (envs.li_ceil + 1)
    dlj = (envs.lj_ceil + 1) if ibase else (envs.li_ceil + envs.lj_ceil + 1)
    nr = envs.nrys_roots
    envs.g_stride_i = nr
    envs.g_stride_k = nr * dli
    envs.g_stride_l = nr * dli * dlk
    envs.g_stride_j = nr * dli * dlk * dll
    envs.g_size     = nr * dli * dlk * dll * dlj
    if kbase:
        envs.g2d_klmax   = envs.g_stride_k
        envs.rx_in_rklrx = envs.rk
        envs.rkrl        = envs.rk - envs.rl
    else:
        envs.g2d_klmax   = envs.g_stride_l
        envs.rx_in_rklrx = envs.rl
        envs.rkrl        = envs.rl - envs.rk
    if ibase:
        envs.g2d_ijmax   = envs.g_stride_i
        envs.rx_in_rijrx = envs.ri
        envs.rirj        = envs.ri - envs.rj
    else:
        envs.g2d_ijmax   = envs.g_stride_j
        envs.rx_in_rijrx = envs.rj
        envs.rirj        = envs.rj - envs.ri
    if kbase:
        envs.f_g0_2d4d = 'ik2d4d' if ibase else 'kj2d4d'
    else:
        envs.f_g0_2d4d = 'il2d4d' if ibase else 'lj2d4d'
    envs.f_g0_2e = 'CINTg0_2e'

def compute_g_index_2e(envs: CINTEnvVars) -> np.ndarray:
    """
    g-array index map for a 2e shell quartet. Returns (nf*3,) int32.
    Loop order: j → l → k → i (matches C default branch for all l).
    """
    di = envs.g_stride_i;  dk = envs.g_stride_k
    dl = envs.g_stride_l;  dj = envs.g_stride_j
    ofx = 0;  ofy = envs.g_size;  ofz = envs.g_size * 2
    i_nx, i_ny, i_nz = cartesian_components(envs.i_l)
    j_nx, j_ny, j_nz = cartesian_components(envs.j_l)
    k_nx, k_ny, k_nz = cartesian_components(envs.k_l)
    l_nx, l_ny, l_nz = cartesian_components(envs.l_l)
    idx = np.empty(envs.nf * 3, dtype=np.int32);  n = 0
    for j in range(envs.nfj):
        for l in range(envs.nfl):
            oflx = ofx + dj * j_nx[j] + dl * l_nx[l]
            ofly = ofy + dj * j_ny[j] + dl * l_ny[l]
            oflz = ofz + dj * j_nz[j] + dl * l_nz[l]
            for k in range(envs.nfk):
                ofkx = oflx + dk * k_nx[k]
                ofky = ofly + dk * k_ny[k]
                ofkz = oflz + dk * k_nz[k]
                for i in range(envs.nfi):
                    idx[n]   = ofkx + di * i_nx[i]
                    idx[n+1] = ofky + di * i_ny[i]
                    idx[n+2] = ofkz + di * i_nz[i]
                    n += 3
    return idx

def compute_pair_data(pairdata: list, ai: np.ndarray, aj: np.ndarray,
                     ri: np.ndarray, rj: np.ndarray,
                     log_maxci: np.ndarray, log_maxcj: np.ndarray,
                     li_ceil: int, lj_ceil: int,
                     iprim: int, jprim: int,
                     rr_ij: float, expcutoff: float) -> bool:
    """
    Precompute shell-pair screening data for all (ip, jp) primitive pairs.
    Populates pairdata in-place (flat list length iprim*jprim, jp outer).
    Returns True if ALL pairs are screened out (empty).
    Uses math.exp to match C libm exactly (avoids numpy SVML 1-ULP difference).
    """
    log_rr_ij = (li_ceil + lj_ceil + 1) * math.log(rr_ij + 1) / 2
    empty = True;  n = 0
    for jp in range(jprim):
        for ip in range(iprim):
            aij   = 1.0 / (ai[ip] + aj[jp])
            eij   = rr_ij * ai[ip] * aj[jp] * aij
            cceij = eij - log_rr_ij - log_maxci[ip] - log_maxcj[jp]
            pdata = pairdata[n];  pdata.cceij = cceij
            if cceij < expcutoff:
                empty = False
                pdata.rij[0] = (ai[ip] * ri[0] + aj[jp] * rj[0]) * aij
                pdata.rij[1] = (ai[ip] * ri[1] + aj[jp] * rj[1]) * aij
                pdata.rij[2] = (ai[ip] * ri[2] + aj[jp] * rj[2]) * aij
                pdata.eij    = math.exp(-eij)
            else:
                pdata.rij[:] = 0.0;  pdata.eij = 0.0
            n += 1
    return empty

def precompute_shell_pairs(opt: CINTOpt, ng: np.ndarray,
                  atm: np.ndarray, natm: int,
                  bas: np.ndarray, nbas: int,
                  env: np.ndarray) -> None:
    """
    Precompute shell-pair data for all (i, j) combinations.

    opt.pairdata[i*nbas+j] is one of:
      list of PairData  — non-negligible pair
      NOVALUE           — screened out
      None              — not set (upper triangle, i<j, handled by symmetry)

    Upper triangle filled as transpose of lower: pairdata[j*nbas+i][ip*jprim+jp]
    == pairdata[i*nbas+j][jp*iprim+ip].
    """
    ecoff = env[PTR_EXPCUTOFF]
    expcutoff = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))
    if opt.log_max_coeff is None:
        set_log_max_coeff(opt, atm, natm, bas, nbas, env)
    tot_prim = int(bas[:nbas, NPRIM_OF].sum())
    if tot_prim == 0 or tot_prim > MAX_PGTO_FOR_PAIRDATA:
        return
    ij_inc   = int(ng[IINC]) + int(ng[JINC])
    kl_inc   = int(ng[KINC]) + int(ng[LINC])
    ijkl_inc = ij_inc if ij_inc > kl_inc else kl_inc
    opt.pairdata = [None] * max(nbas * nbas, 1)
    for i in range(nbas):
        ri        = env[atm[bas[i, ATOM_OF], PTR_COORD] : atm[bas[i, ATOM_OF], PTR_COORD] + 3]
        ai        = env[bas[i, PTR_EXP] : bas[i, PTR_EXP] + bas[i, NPRIM_OF]]
        iprim     = int(bas[i, NPRIM_OF]);  li = int(bas[i, ANG_OF])
        log_maxci = opt.log_max_coeff[i]
        for j in range(i + 1):
            rj        = env[atm[bas[j, ATOM_OF], PTR_COORD] : atm[bas[j, ATOM_OF], PTR_COORD] + 3]
            aj        = env[bas[j, PTR_EXP] : bas[j, PTR_EXP] + bas[j, NPRIM_OF]]
            jprim     = int(bas[j, NPRIM_OF]);  lj = int(bas[j, ANG_OF])
            log_maxcj = opt.log_max_coeff[j]
            rr    = float(np.sum((ri - rj) ** 2))
            pdata = [PairData(rij=np.zeros(3), eij=0.0, cceij=0.0)
                     for _ in range(iprim * jprim)]
            empty = compute_pair_data(pdata, ai, aj, ri, rj, log_maxci, log_maxcj,
                                     li + ijkl_inc, lj, iprim, jprim, rr, expcutoff)
            if i == 0 and j == 0:
                opt.pairdata[0] = pdata
            elif not empty:
                opt.pairdata[i * nbas + j] = pdata
                if i != j:
                    pdata_T = [deepcopy(pdata[jp * iprim + ip])
                               for ip in range(iprim) for jp in range(jprim)]
                    opt.pairdata[j * nbas + i] = pdata_T
            else:
                opt.pairdata[i * nbas + j] = NOVALUE
                opt.pairdata[j * nbas + i] = NOVALUE

def populate_optimizer_2e(opt: CINTOpt, ng: np.ndarray,
                          atm: np.ndarray, natm: int,
                          bas: np.ndarray, nbas: int,
                          env: np.ndarray) -> None:
    """
    Populate all optimizer tables for a 2e integral:
      - pairdata     via precompute_shell_pairs       (includes log_max_coeff)
      - non0coeff    via set_nonzero_coeff
      - index_xyz    via generate_index_xyz with order=4 and init_envvars_2e
    """
    precompute_shell_pairs(opt, ng, atm, natm, bas, nbas, env)
    set_nonzero_coeff(opt, atm, natm, bas, nbas, env)
    generate_index_xyz(opt, init_envvars_2e, compute_g_index_2e,
            4, 0, ng, atm, natm, bas, nbas, env)

def init_envvars_3c2e(envs: CINTEnvVars, ng: np.ndarray, shls: np.ndarray,
                             atm: np.ndarray, natm: int,
                             bas: np.ndarray, nbas: int, env: np.ndarray) -> None:
    """Populate CINTEnvVars for a 3c2e integral over shell triple (i,j,k).
    The 3c2e integral maps k -> ll_ceil position and sets lk_ceil=0 to reuse
    CINTg0_2e_lj2d4d. The 4th shell (l) is a dummy s-function with al=0."""
    envs.natm = natm;  envs.nbas = nbas
    envs.atm  = atm;   envs.bas  = bas;  envs.env = env;  envs.shls = shls
    i_sh, j_sh, k_sh = int(shls[0]), int(shls[1]), int(shls[2])
    envs.i_l = int(bas[i_sh, ANG_OF]);  envs.j_l = int(bas[j_sh, ANG_OF])
    envs.k_l = int(bas[k_sh, ANG_OF]);  envs.l_l = 0
    envs.x_ctr = np.array([bas[i_sh, NCTR_OF], bas[j_sh, NCTR_OF],
                            bas[k_sh, NCTR_OF], 1], dtype=np.int32)
    envs.nfi = (envs.i_l + 1) * (envs.i_l + 2) // 2
    envs.nfj = (envs.j_l + 1) * (envs.j_l + 2) // 2
    envs.nfk = (envs.k_l + 1) * (envs.k_l + 2) // 2
    envs.nfl = 1
    envs.nf  = envs.nfi * envs.nfj * envs.nfk
    i_atom = int(bas[i_sh, ATOM_OF]);  j_atom = int(bas[j_sh, ATOM_OF])
    k_atom = int(bas[k_sh, ATOM_OF])
    envs.ri = env[atm[i_atom, PTR_COORD] : atm[i_atom, PTR_COORD] + 3]
    envs.rj = env[atm[j_atom, PTR_COORD] : atm[j_atom, PTR_COORD] + 3]
    envs.rk = env[atm[k_atom, PTR_COORD] : atm[k_atom, PTR_COORD] + 3]
    envs.rl = envs.rk.copy()  # dummy l = k position
    envs.common_factor = (math.pi ** 3) * 2 / math.sqrt(math.pi) \
        * sph_harmonic_norm(envs.i_l) * sph_harmonic_norm(envs.j_l) \
        * sph_harmonic_norm(envs.k_l)
    ecoff = env[PTR_EXPCUTOFF]
    envs.expcutoff    = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))
    envs.gbits        = int(ng[GSHIFT])
    envs.ncomp_e1     = int(ng[POS_E1])
    envs.ncomp_e2     = int(ng[POS_E2])
    envs.ncomp_tensor = int(ng[TENSOR])
    envs.li_ceil = envs.i_l + int(ng[IINC]);  envs.lj_ceil = envs.j_l + int(ng[JINC])
    envs.lk_ceil = 0  # to reuse CINTg0_2e_lj2d4d
    envs.ll_ceil = envs.k_l + int(ng[KINC])
    envs.nrys_roots = (envs.li_ceil + envs.lj_ceil + envs.ll_ceil) // 2 + 1
    ibase = int(envs.li_ceil > envs.lj_ceil)
    if envs.nrys_roots <= 2:
        ibase = 0
    if ibase:
        dli = envs.li_ceil + envs.lj_ceil + 1
        dlj = envs.lj_ceil + 1
    else:
        dli = envs.li_ceil + 1
        dlj = envs.li_ceil + envs.lj_ceil + 1
    dlk = envs.ll_ceil + 1
    nr = envs.nrys_roots
    envs.g_stride_i = nr
    envs.g_stride_k = nr * dli
    envs.g_stride_l = nr * dli  # same as g_stride_k for 3c2e
    envs.g_stride_j = nr * dli * dlk
    envs.g_size     = nr * dli * dlk * dlj
    envs.al  = 0.0
    envs.rkl = envs.rk.copy()
    envs.rklrx = np.zeros(3)
    envs.rx_in_rklrx = envs.rk
    envs.rkrl = envs.rk.copy()
    envs.g2d_klmax = envs.g_stride_k
    if ibase:
        envs.g2d_ijmax   = envs.g_stride_i
        envs.rx_in_rijrx = envs.ri
        envs.rirj        = envs.ri - envs.rj
        envs.f_g0_2d4d   = 'il2d4d'
    else:
        envs.g2d_ijmax   = envs.g_stride_j
        envs.rx_in_rijrx = envs.rj
        envs.rirj        = envs.rj - envs.ri
        envs.f_g0_2d4d   = 'lj2d4d'
    envs.f_g0_2e = 'CINTg0_2e'


def build_3c2e_optimizer(opt_ref,
                            atm: np.ndarray, natm: int,
                            bas: np.ndarray, nbas: int,
                            env: np.ndarray) -> CINTOpt:
    """3c2e optimizer entry point."""
    ng  = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    opt = create_optimizer(atm, natm, bas, nbas, env)
    precompute_shell_pairs(opt, ng, atm, natm, bas, nbas, env)
    set_nonzero_coeff(opt, atm, natm, bas, nbas, env)
    generate_index_xyz(opt, init_envvars_3c2e, compute_g_index_2e,
            3, 0, ng, atm, natm, bas, nbas, env)
    return opt


def build_2e_optimizer(opt_ref,
                           atm: np.ndarray, natm: int,
                           bas: np.ndarray, nbas: int,
                           env: np.ndarray) -> CINTOpt:
    """
    2e optimizer entry point. Matches C signature:
        void build_2e_optimizer(CINTOpt **opt, ...)
    opt_ref is ignored; Python returns the opt instead of writing through a pointer.
    """
    ng  = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    opt = create_optimizer(atm, natm, bas, nbas, env)
    populate_optimizer_2e(opt, ng, atm, natm, bas, nbas, env)
    return opt
