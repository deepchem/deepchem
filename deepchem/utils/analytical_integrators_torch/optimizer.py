"""
Pure PyTorch port of the libcint optimizer pipeline.

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
import math
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Callable

# Constants
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

# sentinel for screened-out shell pairs
NOVALUE           = object()


# Data structures
class PairData:
    """Shell-pair precomputed data."""
    def __init__(self, rij: torch.Tensor, eij: torch.Tensor, cceij: torch.Tensor):
        self.rij = rij
        self.eij = eij
        self.cceij = cceij


@dataclass
class CINTOpt:
    """Optimizer struct — holds precomputed tables."""
    nbas:            int
    index_xyz_array: Optional[list] = None
    non0ctr:         Optional[list] = None
    sortedidx:       Optional[list] = None
    log_max_coeff:   Optional[list] = None
    pairdata:        Optional[list] = None


@dataclass
class CINTEnvVars:
    """Environment variables for a single shell-pair/quartet integral."""
    atm:   torch.Tensor = None
    bas:   torch.Tensor = None
    env:   torch.Tensor = None
    shls:  torch.Tensor = None

    natm:  int = 0;   nbas:  int = 0

    i_l:   int = 0;   j_l:   int = 0
    k_l:   int = 0;   l_l:   int = 0

    nfi:   int = 0;   nfj:   int = 0
    nfk:   int = 0;   nfl:   int = 0
    nf:    int = 0

    x_ctr: torch.Tensor = None

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

    rirj:  torch.Tensor = None
    rkrl:  torch.Tensor = None

    ri:    torch.Tensor = None
    rj:    torch.Tensor = None
    rk:    torch.Tensor = None
    rl:    torch.Tensor = None

    rx_in_rijrx: torch.Tensor = None
    rx_in_rklrx: torch.Tensor = None

    ai:    float = 0.0;  aj: float = 0.0
    ak:    float = 0.0;  al: float = 0.0

    rij:   torch.Tensor = None
    rijrx: torch.Tensor = None
    aij:   float = 0.0
    rkl:   torch.Tensor = None
    rklrx: torch.Tensor = None
    akl:   float = 0.0

    idx:   torch.Tensor = None

    f_g0_2e:   Optional[str]      = None
    f_g0_2d4d: Optional[str]      = None
    f_gout:    Optional[Callable]  = None


def cartesian_components(lmax: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate Cartesian component exponents (nx, ny, nz) for angular
    momentum lmax, in libcint canonical order.

    Returns
    -------
    nx, ny, nz : each shape ((lmax+1)*(lmax+2)//2,) int32
    """
    nf = (lmax + 1) * (lmax + 2) // 2
    nx = torch.empty(nf, dtype=torch.int32)
    ny = torch.empty(nf, dtype=torch.int32)
    nz = torch.empty(nf, dtype=torch.int32)
    inc = 0
    for lx in range(lmax, -1, -1):
        for ly in range(lmax - lx, -1, -1):
            nx[inc] = lx
            ny[inc] = ly
            nz[inc] = lmax - lx - ly
            inc += 1
    return nx, ny, nz


def compute_log_max_coeff(log_maxc: torch.Tensor, coeff: torch.Tensor,
                           nprim: int, ictr: int) -> None:
    """Compute approx_log of max |coeff| over contractions for each primitive."""
    c = coeff.reshape(ictr, nprim)
    maxc = c.abs().amax(dim=0)
    log_maxc[:] = torch.log(maxc + 1e-300) + 1.0


def compute_log_max_coeffs(bas: torch.Tensor, nbas: int,
                           env: torch.Tensor) -> Optional[list]:
    """Compute approximate log of max absolute contraction coefficient per primitive."""
    nprims = bas[:nbas, NPRIM_OF].to(torch.int64)
    tot_prim = int(nprims.sum().item())
    if tot_prim == 0:
        return None

    plog_maxc = torch.empty(tot_prim, dtype=torch.float64)
    result = []
    offset = 0

    for i in range(nbas):
        ip = int(nprims[i].item())
        ic = int(bas[i, NCTR_OF].item())
        ci = env[int(bas[i, PTR_COEFF].item()):int(bas[i, PTR_COEFF].item()) + ip * ic]
        slc = plog_maxc[offset:offset + ip]
        compute_log_max_coeff(slc, ci, ip, ic)
        result.append(slc)
        offset += ip

    return result


def nonzero_coeff_by_shell(ci: torch.Tensor, iprim: int,
                           ictr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition contraction indices so non-zero coefficients come first."""
    c = ci.reshape(ictr, iprim)
    mask = (c != 0)
    order = torch.argsort(~mask, dim=0, stable=True)
    non0ctr = mask.sum(dim=0).to(torch.int32)
    sortedidx = order.T.to(torch.int32)
    return sortedidx, non0ctr


def compute_nonzero_coeffs(bas: torch.Tensor, nbas: int,
                           env: torch.Tensor) -> tuple[Optional[list], Optional[list]]:
    """Compute non-zero contraction indices and counts for all shells."""
    nprims = bas[:nbas, NPRIM_OF].to(torch.int64)
    nctrs = bas[:nbas, NCTR_OF].to(torch.int64)
    tot_prim = int(nprims.sum().item())
    tot_prim_ctr = int((nprims * nctrs).sum().item())
    if tot_prim == 0:
        return None, None

    pnon0ctr = torch.empty(tot_prim, dtype=torch.int32)
    psortedidx = torch.empty(tot_prim_ctr, dtype=torch.int32)

    non0ctr_list = []
    sortedidx_list = []
    p0 = pc0 = 0

    for i in range(nbas):
        ip, ic = int(nprims[i].item()), int(nctrs[i].item())
        ci = env[int(bas[i, PTR_COEFF].item()):int(bas[i, PTR_COEFF].item()) + ip * ic]
        sidx, n0 = nonzero_coeff_by_shell(ci, ip, ic)

        pnon0ctr[p0:p0 + ip] = n0
        psortedidx[pc0:pc0 + ip * ic] = sidx.ravel()

        non0ctr_list.append(pnon0ctr[p0:p0 + ip])
        sortedidx_list.append(psortedidx[pc0:pc0 + ip * ic].reshape(ip, ic))

        p0 += ip
        pc0 += ip * ic

    return non0ctr_list, sortedidx_list


def make_fake_basis(bas: torch.Tensor, nbas: int) -> tuple[torch.Tensor, int]:
    """Build a minimal fake basis with one shell per angular momentum 0..max_l."""
    max_l = int(bas[:nbas, ANG_OF].max().item())
    fakebas = torch.zeros((max_l + 1, BAS_SLOTS), dtype=torch.int32)
    for i in range(max_l + 1):
        fakebas[i, ANG_OF] = i
    return fakebas, max_l


FAC_SP = [0.282094791773878, 0.488602511902920]


def sph_harmonic_norm(l: int) -> float:
    """Spherical harmonic normalisation for s/p shells; 1.0 for l >= 2."""
    if l == 0: return FAC_SP[0]
    if l == 1: return FAC_SP[1]
    return 1.0


def init_envvars_1e(envs: CINTEnvVars, ng: torch.Tensor,
                    shls: torch.Tensor,
                    atm: torch.Tensor, natm: int,
                    bas: torch.Tensor, nbas: int,
                    env: torch.Tensor) -> None:
    """Populate CINTEnvVars for a 1e integral over shell pair."""
    envs.natm = natm;  envs.nbas = nbas
    envs.atm = atm;    envs.bas = bas
    envs.env = env;    envs.shls = shls

    i_sh, j_sh = int(shls[0].item()), int(shls[1].item())
    envs.i_l = int(bas[i_sh, ANG_OF].item())
    envs.j_l = int(bas[j_sh, ANG_OF].item())

    envs.x_ctr = torch.tensor([bas[i_sh, NCTR_OF], bas[j_sh, NCTR_OF], 0, 0],
                               dtype=torch.int32)
    envs.nfi = (envs.i_l + 1) * (envs.i_l + 2) // 2
    envs.nfj = (envs.j_l + 1) * (envs.j_l + 2) // 2
    envs.nf = envs.nfi * envs.nfj

    i_atom = int(bas[i_sh, ATOM_OF].item())
    j_atom = int(bas[j_sh, ATOM_OF].item())
    envs.ri = env[int(atm[i_atom, PTR_COORD].item()):int(atm[i_atom, PTR_COORD].item()) + 3]
    envs.rj = env[int(atm[j_atom, PTR_COORD].item()):int(atm[j_atom, PTR_COORD].item()) + 3]

    envs.common_factor = 1.0
    ecoff = env[PTR_EXPCUTOFF].item()
    envs.expcutoff = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))

    envs.gbits        = int(ng[GSHIFT].item())
    envs.ncomp_e1     = int(ng[POS_E1].item())
    envs.ncomp_tensor = int(ng[TENSOR].item())

    envs.li_ceil    = envs.i_l + int(ng[IINC].item())
    envs.lj_ceil    = envs.j_l + int(ng[JINC].item())
    envs.nrys_roots = (envs.li_ceil + envs.lj_ceil) // 2 + 1

    dli             = envs.li_ceil + envs.lj_ceil + 1
    envs.g_stride_i = 1
    envs.g_stride_j = dli
    envs.g_size     = dli * (envs.lj_ceil + 1)


def compute_g_index_1e(envs: CINTEnvVars) -> torch.Tensor:
    """Compute the g-array index map for a 1e shell pair. Returns (nf*3,) int32."""
    di = envs.g_stride_i;  dj = envs.g_stride_j
    ofx = 0;  ofy = envs.g_size;  ofz = envs.g_size * 2

    i_nx, i_ny, i_nz = cartesian_components(envs.i_l)
    j_nx, j_ny, j_nz = cartesian_components(envs.j_l)

    idx = torch.empty(envs.nf * 3, dtype=torch.int32)
    n = 0
    for j in range(envs.nfj):
        ofjx = ofx + dj * int(j_nx[j].item())
        ofjy = ofy + dj * int(j_ny[j].item())
        ofjz = ofz + dj * int(j_nz[j].item())
        for i in range(envs.nfi):
            idx[n]   = ofjx + di * int(i_nx[i].item())
            idx[n+1] = ofjy + di * int(i_ny[i].item())
            idx[n+2] = ofjz + di * int(i_nz[i].item())
            n += 3
    return idx


def generate_index_xyz(finit, findex_xyz,
                       order: int, max_l_override: int,
                       ng: torch.Tensor,
                       atm: torch.Tensor, natm: int,
                       bas: torch.Tensor, nbas: int,
                       env: torch.Tensor) -> list:
    """Pre-compute g-array index maps for all shell-pair angular momentum combinations."""
    fakebas, max_l1 = make_fake_basis(bas, nbas)
    max_l = max_l1 if max_l_override == 0 else min(max_l_override, max_l1)
    fakenbas = max_l + 1

    from itertools import product as _prod
    index_xyz_array = [None] * ((max_l + 1) * (LMAX1 ** (order - 1)))
    envs = CINTEnvVars()
    rng = range(max_l + 1)

    for combo in _prod(rng, repeat=order):
        shls = torch.tensor(list(combo) + [0] * (4 - order), dtype=torch.int32)
        finit(envs, ng, shls, atm, natm, fakebas, fakenbas, env)
        flat_idx = sum(c * LMAX1 ** (order - 1 - p) for p, c in enumerate(combo))
        index_xyz_array[flat_idx] = findex_xyz(envs)

    return index_xyz_array


def build_optimizer_1e(ng_list, atm, natm, bas, nbas, env):
    """Common 1e optimizer builder."""
    ng = torch.tensor(ng_list, dtype=torch.int32)
    log_max_coeff = compute_log_max_coeffs(bas, nbas, env)
    non0ctr, sortedidx = compute_nonzero_coeffs(bas, nbas, env)
    index_xyz_array = generate_index_xyz(
        init_envvars_1e, compute_g_index_1e,
        2, 0, ng, atm, natm, bas, nbas, env)
    return CINTOpt(
        nbas=nbas,
        log_max_coeff=log_max_coeff,
        non0ctr=non0ctr,
        sortedidx=sortedidx,
        index_xyz_array=index_xyz_array,
    )


def build_overlap_optimizer(opt_ref, atm, natm, bas, nbas, env):
    return build_optimizer_1e([0, 0, 0, 0, 0, 1, 1, 1], atm, natm, bas, nbas, env)


def build_kinetic_optimizer(opt_ref, atm, natm, bas, nbas, env):
    return build_optimizer_1e([0, 2, 0, 0, 2, 1, 1, 1], atm, natm, bas, nbas, env)


def build_nuclear_optimizer(opt_ref, atm, natm, bas, nbas, env):
    return build_optimizer_1e([0, 0, 0, 0, 0, 1, 0, 1], atm, natm, bas, nbas, env)


def init_envvars_2e(envs: CINTEnvVars, ng: torch.Tensor, shls: torch.Tensor,
                    atm: torch.Tensor, natm: int,
                    bas: torch.Tensor, nbas: int, env: torch.Tensor) -> None:
    """Populate CINTEnvVars for a 2e integral over shell quartet (i,j,k,l)."""
    envs.natm = natm;  envs.nbas = nbas
    envs.atm = atm;    envs.bas = bas;  envs.env = env;  envs.shls = shls
    i_sh = int(shls[0].item());  j_sh = int(shls[1].item())
    k_sh = int(shls[2].item());  l_sh = int(shls[3].item())
    envs.i_l = int(bas[i_sh, ANG_OF].item());  envs.j_l = int(bas[j_sh, ANG_OF].item())
    envs.k_l = int(bas[k_sh, ANG_OF].item());  envs.l_l = int(bas[l_sh, ANG_OF].item())
    envs.x_ctr = torch.tensor([bas[i_sh, NCTR_OF], bas[j_sh, NCTR_OF],
                                bas[k_sh, NCTR_OF], bas[l_sh, NCTR_OF]], dtype=torch.int32)
    envs.nfi = (envs.i_l + 1) * (envs.i_l + 2) // 2
    envs.nfj = (envs.j_l + 1) * (envs.j_l + 2) // 2
    envs.nfk = (envs.k_l + 1) * (envs.k_l + 2) // 2
    envs.nfl = (envs.l_l + 1) * (envs.l_l + 2) // 2
    envs.nf = envs.nfi * envs.nfj * envs.nfk * envs.nfl
    i_atom = int(bas[i_sh, ATOM_OF].item());  j_atom = int(bas[j_sh, ATOM_OF].item())
    k_atom = int(bas[k_sh, ATOM_OF].item());  l_atom = int(bas[l_sh, ATOM_OF].item())
    envs.ri = env[int(atm[i_atom, PTR_COORD].item()):int(atm[i_atom, PTR_COORD].item()) + 3]
    envs.rj = env[int(atm[j_atom, PTR_COORD].item()):int(atm[j_atom, PTR_COORD].item()) + 3]
    envs.rk = env[int(atm[k_atom, PTR_COORD].item()):int(atm[k_atom, PTR_COORD].item()) + 3]
    envs.rl = env[int(atm[l_atom, PTR_COORD].item()):int(atm[l_atom, PTR_COORD].item()) + 3]
    envs.common_factor = (math.pi ** 3) * 2 / math.sqrt(math.pi) \
        * sph_harmonic_norm(envs.i_l) * sph_harmonic_norm(envs.j_l) \
        * sph_harmonic_norm(envs.k_l) * sph_harmonic_norm(envs.l_l)
    ecoff = env[PTR_EXPCUTOFF].item()
    envs.expcutoff = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))
    envs.gbits        = int(ng[GSHIFT].item())
    envs.ncomp_e1     = int(ng[POS_E1].item())
    envs.ncomp_e2     = int(ng[POS_E2].item())
    envs.ncomp_tensor = int(ng[TENSOR].item())
    envs.li_ceil = envs.i_l + int(ng[IINC].item());  envs.lj_ceil = envs.j_l + int(ng[JINC].item())
    envs.lk_ceil = envs.k_l + int(ng[KINC].item());  envs.ll_ceil = envs.l_l + int(ng[LINC].item())
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


def compute_g_index_2e(envs: CINTEnvVars) -> torch.Tensor:
    """g-array index map for a 2e shell quartet. Returns (nf*3,) int32."""
    di = envs.g_stride_i;  dk = envs.g_stride_k
    dl = envs.g_stride_l;  dj = envs.g_stride_j
    ofx = 0;  ofy = envs.g_size;  ofz = envs.g_size * 2
    i_nx, i_ny, i_nz = cartesian_components(envs.i_l)
    j_nx, j_ny, j_nz = cartesian_components(envs.j_l)
    k_nx, k_ny, k_nz = cartesian_components(envs.k_l)
    l_nx, l_ny, l_nz = cartesian_components(envs.l_l)
    idx = torch.empty(envs.nf * 3, dtype=torch.int32);  n = 0
    for j in range(envs.nfj):
        for l in range(envs.nfl):
            oflx = ofx + dj * int(j_nx[j].item()) + dl * int(l_nx[l].item())
            ofly = ofy + dj * int(j_ny[j].item()) + dl * int(l_ny[l].item())
            oflz = ofz + dj * int(j_nz[j].item()) + dl * int(l_nz[l].item())
            for k in range(envs.nfk):
                ofkx = oflx + dk * int(k_nx[k].item())
                ofky = ofly + dk * int(k_ny[k].item())
                ofkz = oflz + dk * int(k_nz[k].item())
                for i in range(envs.nfi):
                    idx[n]   = ofkx + di * int(i_nx[i].item())
                    idx[n+1] = ofky + di * int(i_ny[i].item())
                    idx[n+2] = ofkz + di * int(i_nz[i].item())
                    n += 3
    return idx


def compute_pair_data(ai: torch.Tensor, aj: torch.Tensor,
                      ri: torch.Tensor, rj: torch.Tensor,
                      log_maxci: torch.Tensor, log_maxcj: torch.Tensor,
                      li_ceil: int, lj_ceil: int,
                      iprim: int, jprim: int,
                      rr_ij: float, expcutoff: float) -> tuple[list, bool]:
    """Compute shell-pair screening data for all (ip, jp) primitive pairs."""
    log_rr_ij = (li_ceil + lj_ceil + 1) * math.log(rr_ij + 1) / 2
    pairdata = []
    empty = True
    for jp in range(jprim):
        for ip in range(iprim):
            aij = 1.0 / (ai[ip] + aj[jp])
            eij = rr_ij * ai[ip] * aj[jp] * aij
            cceij = eij - log_rr_ij - log_maxci[ip] - log_maxcj[jp]
            if cceij.item() < expcutoff:
                empty = False
                rij = torch.stack([
                    (ai[ip] * ri[0] + aj[jp] * rj[0]) * aij,
                    (ai[ip] * ri[1] + aj[jp] * rj[1]) * aij,
                    (ai[ip] * ri[2] + aj[jp] * rj[2]) * aij,
                ])
                pairdata.append(PairData(rij=rij, eij=torch.exp(-eij), cceij=cceij))
            else:
                pairdata.append(PairData(rij=torch.zeros(3, dtype=torch.float64),
                                         eij=torch.tensor(0.0, dtype=torch.float64),
                                         cceij=cceij))
    return pairdata, empty


def precompute_shell_pairs(ng: torch.Tensor, log_max_coeff: list,
                           atm: torch.Tensor,
                           bas: torch.Tensor, nbas: int,
                           env: torch.Tensor) -> Optional[list]:
    """Compute shell-pair data for all (i, j) combinations."""
    ecoff = env[PTR_EXPCUTOFF].item()
    expcutoff = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))
    tot_prim = int(bas[:nbas, NPRIM_OF].sum().item())
    if tot_prim == 0 or tot_prim > MAX_PGTO_FOR_PAIRDATA:
        return None
    ij_inc = int(ng[IINC].item()) + int(ng[JINC].item())
    kl_inc = int(ng[KINC].item()) + int(ng[LINC].item())
    ijkl_inc = ij_inc if ij_inc > kl_inc else kl_inc
    pairdata = [None] * max(nbas * nbas, 1)
    for i in range(nbas):
        ri = env[int(atm[int(bas[i, ATOM_OF].item()), PTR_COORD].item()):
                 int(atm[int(bas[i, ATOM_OF].item()), PTR_COORD].item()) + 3]
        ai = env[int(bas[i, PTR_EXP].item()):int(bas[i, PTR_EXP].item()) + int(bas[i, NPRIM_OF].item())]
        iprim = int(bas[i, NPRIM_OF].item());  li = int(bas[i, ANG_OF].item())
        log_maxci = log_max_coeff[i]
        for j in range(i + 1):
            rj = env[int(atm[int(bas[j, ATOM_OF].item()), PTR_COORD].item()):
                     int(atm[int(bas[j, ATOM_OF].item()), PTR_COORD].item()) + 3]
            aj = env[int(bas[j, PTR_EXP].item()):int(bas[j, PTR_EXP].item()) + int(bas[j, NPRIM_OF].item())]
            jprim = int(bas[j, NPRIM_OF].item());  lj = int(bas[j, ANG_OF].item())
            log_maxcj = log_max_coeff[j]
            rr = ((ri - rj) ** 2).sum().item()
            pdata, empty = compute_pair_data(
                ai, aj, ri, rj, log_maxci, log_maxcj,
                li + ijkl_inc, lj, iprim, jprim, rr, expcutoff)
            if i == 0 and j == 0:
                pairdata[0] = pdata
            elif not empty:
                pairdata[i * nbas + j] = pdata
                if i != j:
                    pdata_T = [deepcopy(pdata[jp * iprim + ip])
                               for ip in range(iprim) for jp in range(jprim)]
                    pairdata[j * nbas + i] = pdata_T
            else:
                pairdata[i * nbas + j] = NOVALUE
                pairdata[j * nbas + i] = NOVALUE
    return pairdata


def build_optimizer_2e(ng: torch.Tensor,
                        atm: torch.Tensor, natm: int,
                        bas: torch.Tensor, nbas: int,
                        env: torch.Tensor) -> CINTOpt:
    """Build a fully populated 2e optimizer."""
    log_max_coeff = compute_log_max_coeffs(bas, nbas, env)
    pairdata = None
    if log_max_coeff is not None:
        pairdata = precompute_shell_pairs(
            ng, log_max_coeff, atm, bas, nbas, env)
    non0ctr, sortedidx = compute_nonzero_coeffs(bas, nbas, env)
    index_xyz_array = generate_index_xyz(
        init_envvars_2e, compute_g_index_2e,
        4, 0, ng, atm, natm, bas, nbas, env)
    return CINTOpt(
        nbas=nbas,
        log_max_coeff=log_max_coeff,
        pairdata=pairdata,
        non0ctr=non0ctr,
        sortedidx=sortedidx,
        index_xyz_array=index_xyz_array,
    )


def init_envvars_3c2e(envs: CINTEnvVars, ng: torch.Tensor, shls: torch.Tensor,
                      atm: torch.Tensor, natm: int,
                      bas: torch.Tensor, nbas: int, env: torch.Tensor) -> None:
    """Populate CINTEnvVars for a 3c2e integral over shell triple (i,j,k)."""
    envs.natm = natm;  envs.nbas = nbas
    envs.atm = atm;    envs.bas = bas;  envs.env = env;  envs.shls = shls
    i_sh = int(shls[0].item());  j_sh = int(shls[1].item());  k_sh = int(shls[2].item())
    envs.i_l = int(bas[i_sh, ANG_OF].item());  envs.j_l = int(bas[j_sh, ANG_OF].item())
    envs.k_l = int(bas[k_sh, ANG_OF].item());  envs.l_l = 0
    envs.x_ctr = torch.tensor([bas[i_sh, NCTR_OF], bas[j_sh, NCTR_OF],
                                bas[k_sh, NCTR_OF], 1], dtype=torch.int32)
    envs.nfi = (envs.i_l + 1) * (envs.i_l + 2) // 2
    envs.nfj = (envs.j_l + 1) * (envs.j_l + 2) // 2
    envs.nfk = (envs.k_l + 1) * (envs.k_l + 2) // 2
    envs.nfl = 1
    envs.nf = envs.nfi * envs.nfj * envs.nfk
    i_atom = int(bas[i_sh, ATOM_OF].item());  j_atom = int(bas[j_sh, ATOM_OF].item())
    k_atom = int(bas[k_sh, ATOM_OF].item())
    envs.ri = env[int(atm[i_atom, PTR_COORD].item()):int(atm[i_atom, PTR_COORD].item()) + 3]
    envs.rj = env[int(atm[j_atom, PTR_COORD].item()):int(atm[j_atom, PTR_COORD].item()) + 3]
    envs.rk = env[int(atm[k_atom, PTR_COORD].item()):int(atm[k_atom, PTR_COORD].item()) + 3]
    envs.rl = envs.rk.clone()
    envs.common_factor = (math.pi ** 3) * 2 / math.sqrt(math.pi) \
        * sph_harmonic_norm(envs.i_l) * sph_harmonic_norm(envs.j_l) \
        * sph_harmonic_norm(envs.k_l)
    ecoff = env[PTR_EXPCUTOFF].item()
    envs.expcutoff = EXPCUTOFF if ecoff == 0 else max(MIN_EXPCUTOFF, float(ecoff))
    envs.gbits        = int(ng[GSHIFT].item())
    envs.ncomp_e1     = int(ng[POS_E1].item())
    envs.ncomp_e2     = int(ng[POS_E2].item())
    envs.ncomp_tensor = int(ng[TENSOR].item())
    envs.li_ceil = envs.i_l + int(ng[IINC].item());  envs.lj_ceil = envs.j_l + int(ng[JINC].item())
    envs.lk_ceil = 0
    envs.ll_ceil = envs.k_l + int(ng[KINC].item())
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
    envs.g_stride_l = nr * dli
    envs.g_stride_j = nr * dli * dlk
    envs.g_size     = nr * dli * dlk * dlj
    envs.al = 0.0
    envs.rkl = envs.rk.clone()
    envs.rklrx = torch.zeros(3, dtype=env.dtype, device=env.device)
    envs.rx_in_rklrx = envs.rk
    envs.rkrl = envs.rk.clone()
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
                         atm: torch.Tensor, natm: int,
                         bas: torch.Tensor, nbas: int,
                         env: torch.Tensor) -> CINTOpt:
    """3c2e optimizer entry point."""
    ng = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
    log_max_coeff = compute_log_max_coeffs(bas, nbas, env)
    pairdata = None
    if log_max_coeff is not None:
        pairdata = precompute_shell_pairs(
            ng, log_max_coeff, atm, bas, nbas, env)
    non0ctr, sortedidx = compute_nonzero_coeffs(bas, nbas, env)
    index_xyz_array = generate_index_xyz(
        init_envvars_3c2e, compute_g_index_2e,
        3, 0, ng, atm, natm, bas, nbas, env)
    return CINTOpt(
        nbas=nbas,
        log_max_coeff=log_max_coeff,
        pairdata=pairdata,
        non0ctr=non0ctr,
        sortedidx=sortedidx,
        index_xyz_array=index_xyz_array,
    )


def build_2e_optimizer(opt_ref,
                       atm: torch.Tensor, natm: int,
                       bas: torch.Tensor, nbas: int,
                       env: torch.Tensor) -> CINTOpt:
    """2e optimizer entry point."""
    ng = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
    return build_optimizer_2e(ng, atm, natm, bas, nbas, env)
