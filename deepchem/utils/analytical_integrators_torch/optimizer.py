"""
Pure PyTorch port of the libcint optimizer pipeline.

Replicates the shell-pair precomputation, exponential screening, and
g-array index generation that libcint performs in its C optimizer structs.
The public API mirrors libcint's ``CINTOpt`` workflow so that the same
driver logic can be used for 1e, 2e, and 3c2e integrals.

References
----------
.. [1] Q. Sun, "Libcint: An efficient general integral library for
   Gaussian basis functions." Journal of Computational Chemistry,
   36(22), 1664–1671 (2015).
.. [2] S. Obara, A. Saika, "Efficient recursive computation of molecular
   integrals over Cartesian Gaussian functions." Journal of Chemical
   Physics, 84(7), 3963–3974 (1986).

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
LMAX1 = 16
BAS_SLOTS = 8
ATM_SLOTS = 6

# bas column indices
ATOM_OF = 0
ANG_OF = 1
NPRIM_OF = 2
NCTR_OF = 3
PTR_EXP = 5
PTR_COEFF = 6

# atm column indices
PTR_COORD = 1

# env index
PTR_EXPCUTOFF = 0

# ng (integral type descriptor) indices
IINC = 0
JINC = 1
KINC = 2
LINC = 3
GSHIFT = 4
POS_E1 = 5
POS_E2 = 6
TENSOR = 7

# cutoff defaults
EXPCUTOFF = 60.0
MIN_EXPCUTOFF = 20.0

# pairdata screening limit
MAX_PGTO_FOR_PAIRDATA = 50000

# math constants
SQRTPI = math.sqrt(math.pi)

# sentinel for screened-out shell pairs
NOVALUE = object()


# Data structures
class PairData:
    """Precomputed data for a single primitive shell pair (ip, jp).

    Stores the weighted center-of-charge position, the exponential
    screening factor, and the combined exponent cutoff value used by
    the Schwarz/distance screening in libcint.

    Parameters
    ----------
    rij : torch.Tensor, shape (3,)
        Weighted center-of-charge: (a_i * R_i + a_j * R_j) / (a_i + a_j).
    eij : torch.Tensor, scalar
        exp(-a_i * a_j / (a_i + a_j) * |R_i - R_j|^2).
    cceij : torch.Tensor, scalar
        Combined exponent cutoff value used for Schwarz screening.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
    def __init__(self, rij: torch.Tensor, eij: torch.Tensor, cceij: torch.Tensor):
        self.rij = rij
        self.eij = eij
        self.cceij = cceij


@dataclass
class CINTOpt:
    """Optimizer struct that holds all precomputed screening tables.

    Mirrors the libcint ``CINTOpt`` C struct.  Built once per integral type
    and reused across all shell pairs/quartets in an integral contraction.

    Parameters
    ----------
    nbas : int
        Number of basis shells.
    index_xyz_array : list, optional
        Pre-computed g-array index maps for every (l_i, l_j, ...) combo.
    non0ctr : list, optional
        Per-shell list of non-zero contraction counts for each primitive.
    sortedidx : list, optional
        Per-shell sorted indices placing non-zero contractions first.
    log_max_coeff : list, optional
        Per-shell log of maximum absolute contraction coefficient.
    pairdata : list, optional
        Flat list of PairData objects, indexed by (i * nbas + j).

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
    nbas:            int
    index_xyz_array: Optional[list] = None
    non0ctr:         Optional[list] = None
    sortedidx:       Optional[list] = None
    log_max_coeff:   Optional[list] = None
    pairdata:        Optional[list] = None


@dataclass
class CINTEnvVars:
    """Environment variables for a single shell-pair or shell-quartet integral.

    Populated by ``init_envvars_1e``, ``init_envvars_2e``, or
    ``init_envvars_3c2e`` before the primitive loop begins.  Mirrors the
    ``CINTEnvVars`` C struct in libcint.

    Parameters
    ----------
    atm, bas, env : torch.Tensor
        Atom, basis, and environment arrays as defined in the libcint
        data layout.
    shls : torch.Tensor, shape (4,) int32
        Shell indices (i, j, k, l) for the current quartet.
    natm, nbas : int
        Number of atoms / shells.
    i_l, j_l, k_l, l_l : int
        Angular momenta of the four shells.
    nfi, nfj, nfk, nfl : int
        Number of Cartesian components for each shell.
    nf : int
        Total number of Cartesian components in the shell block.
    x_ctr : torch.Tensor, shape (4,) int32
        Contraction lengths for each shell.
    gbits : int
        Bit-field controlling g-array layout (GSHIFT).
    ncomp_e1, ncomp_e2, ncomp_tensor : int
        Number of components for electron 1, electron 2, and tensor index.
    li_ceil, lj_ceil, lk_ceil, ll_ceil : int
        Effective angular momenta including derivative increments.
    g_stride_i/k/l/j : int
        Strides through the g-array for each angular index.
    nrys_roots : int
        Number of Rys quadrature points required.
    g_size : int
        Size of one x/y/z strip of the g-array.
    g2d_ijmax, g2d_klmax : int
        2D recurrence strides for ij and kl pairs.
    common_factor : float
        Global prefactor for the integral (pi factors, norms).
    expcutoff : float
        Exponent screening threshold.
    rirj, rkrl : torch.Tensor
        Vectors R_i - R_j and R_k - R_l.
    ri, rj, rk, rl : torch.Tensor
        Atom positions for the four shells.
    rx_in_rijrx, rx_in_rklrx : torch.Tensor
        Expansion center for ij and kl 2D recurrence.
    ai, aj, ak, al : float
        Current primitive exponents.
    rij, rijrx : torch.Tensor
        Gaussian product center and its offset from the expansion center.
    aij : float
        Sum of ij primitive exponents.
    rkl, rklrx : torch.Tensor
        Gaussian product center (kl) and its offset.
    akl : float
        Sum of kl primitive exponents.
    idx : torch.Tensor
        g-array index map for the current shell block.
    f_g0_2e : str, optional
        Tag for the 2e g-value generator variant.
    f_g0_2d4d : str, optional
        Tag for the 2D→4D recurrence variant.
    f_gout : callable, optional
        Function to extract the contracted integral from the g-array.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Generate Cartesian component exponents for angular momentum ``lmax``.

    Returns the (nx, ny, nz) triples in libcint canonical order:
    decreasing lx, then decreasing ly, nz = lmax - lx - ly.

    Parameters
    ----------
    lmax : int
        Angular momentum quantum number (0 = s, 1 = p, 2 = d, ...).

    Returns
    -------
    nx : torch.Tensor, shape (nf,) int32
        x-exponents.
    ny : torch.Tensor, shape (nf,) int32
        y-exponents.
    nz : torch.Tensor, shape (nf,) int32
        z-exponents.
        nf = (lmax+1)*(lmax+2)//2.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
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
    """Compute log of the maximum absolute contraction coefficient per primitive.

    For each primitive p, writes log(max_c |C_{c,p}|) + 1 into
    ``log_maxc[p]``.  The +1 offset matches libcint's convention and
    ensures a small positive value even when all coefficients are zero.
    Used for Schwarz exponential screening.

    Parameters
    ----------
    log_maxc : torch.Tensor, shape (nprim,) float64
        Output buffer; modified in-place.
    coeff : torch.Tensor, shape (ictr * nprim,) float64
        Flattened contraction coefficients in row-major (nctr, nprim) order.
    nprim : int
        Number of primitive Gaussians for this shell.
    ictr : int
        Number of contractions for this shell.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
    c = coeff.reshape(ictr, nprim)
    maxc = c.abs().amax(dim=0)
    log_maxc[:] = torch.log(maxc + 1e-300) + 1.0


def compute_log_max_coeffs(bas: torch.Tensor, nbas: int,
                           env: torch.Tensor) -> Optional[list]:
    """Compute log of max absolute contraction coefficient for every shell.

    Calls :func:`compute_log_max_coeff` for each shell and accumulates the
    results in a shared flat buffer, returning a list of views into that
    buffer (one per shell).

    Parameters
    ----------
    bas : torch.Tensor, shape (nbas, BAS_SLOTS) int32
        Basis-shell array in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor, shape (nenv,) float64
        Environment array containing exponents and coefficients.

    Returns
    -------
    list of torch.Tensor or None
        List of length ``nbas`` where each element is a 1-D float64 tensor
        of shape ``(nprim_i,)`` containing the log-max-coeff values.
        Returns ``None`` if the basis has no primitives.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Partition primitive indices so non-zero contractions come first.

    Returns a stable sorted index array and a per-primitive count of
    non-zero contractions.  Putting non-zero coefficients first lets the
    inner contraction loop skip zero-coefficient primitives.

    Parameters
    ----------
    ci : torch.Tensor, shape (ictr * iprim,) float64
        Flattened contraction coefficients in row-major (ictr, iprim) order.
    iprim : int
        Number of primitive Gaussians for this shell.
    ictr : int
        Number of contracted Gaussians for this shell.

    Returns
    -------
    sortedidx : torch.Tensor, shape (iprim, ictr) int32
        For each primitive p and contraction c, sortedidx[p, c] is the
        contraction index reordered so that non-zero entries come first.
    non0ctr : torch.Tensor, shape (iprim,) int32
        Number of non-zero contractions for each primitive.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
    c = ci.reshape(ictr, iprim)
    mask = (c != 0)
    order = torch.argsort(~mask, dim=0, stable=True)
    non0ctr = mask.sum(dim=0).to(torch.int32)
    sortedidx = order.T.to(torch.int32)
    return sortedidx, non0ctr


def compute_nonzero_coeffs(bas: torch.Tensor, nbas: int,
                           env: torch.Tensor) -> tuple[Optional[list], Optional[list]]:
    """Compute non-zero contraction indices and counts for every shell.

    Calls :func:`nonzero_coeff_by_shell` for each shell and packs the
    results into shared flat buffers, returning list-of-views for both
    ``non0ctr`` and ``sortedidx``.

    Parameters
    ----------
    bas : torch.Tensor, shape (nbas, BAS_SLOTS) int32
        Basis-shell array in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor, shape (nenv,) float64
        Environment array containing contraction coefficients.

    Returns
    -------
    non0ctr_list : list of torch.Tensor or None
        Per-shell non-zero contraction counts.  ``None`` if no primitives.
    sortedidx_list : list of torch.Tensor or None
        Per-shell sorted primitive indices.  ``None`` if no primitives.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Build a minimal fake basis with one shell per angular momentum up to max_l.

    Used by :func:`generate_index_xyz` to pre-compute g-array index maps
    for every angular-momentum combination without iterating over the full
    basis.

    Parameters
    ----------
    bas : torch.Tensor, shape (nbas, BAS_SLOTS) int32
        Actual basis-shell array.
    nbas : int
        Number of actual basis shells.

    Returns
    -------
    fakebas : torch.Tensor, shape (max_l+1, BAS_SLOTS) int32
        Fake basis with one shell per l = 0 .. max_l.
    max_l : int
        Highest angular momentum present in ``bas``.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
    max_l = int(bas[:nbas, ANG_OF].max().item())
    fakebas = torch.zeros((max_l + 1, BAS_SLOTS), dtype=torch.int32)
    for i in range(max_l + 1):
        fakebas[i, ANG_OF] = i
    return fakebas, max_l


FAC_SP = [0.282094791773878, 0.488602511902920]


def sph_harmonic_norm(l: int) -> float:
    """Return the spherical harmonic normalisation factor for shells l = 0 or 1.

    Returns the prefactor that converts raw Cartesian GTO values to the
    normalised real solid-harmonic basis used by libcint.  For l >= 2 the
    normalisation is absorbed into the contraction coefficients and this
    function returns 1.0.

    Parameters
    ----------
    l : int
        Angular momentum of the shell (0 = s, 1 = p, >= 2 → 1.0).

    Returns
    -------
    float
        1 / sqrt(4*pi) for l=0, sqrt(3 / (4*pi)) for l=1, 1.0 otherwise.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
    if l == 0: return FAC_SP[0]
    if l == 1: return FAC_SP[1]
    return 1.0


def init_envvars_1e(envs: CINTEnvVars, ng: torch.Tensor,
                    shls: torch.Tensor,
                    atm: torch.Tensor, natm: int,
                    bas: torch.Tensor, nbas: int,
                    env: torch.Tensor) -> None:
    """Populate a CINTEnvVars struct for a one-electron integral over a shell pair.

    Sets angular momenta, contraction counts, atom positions, g-array
    strides, and the number of Rys roots required for the (i_sh, j_sh)
    shell pair described by the integral type descriptor ``ng``.

    Parameters
    ----------
    envs : CINTEnvVars
        Output struct populated in-place.
    ng : torch.Tensor, shape (8,) int32
        Integral type descriptor [IINC, JINC, KINC, LINC, GSHIFT,
        POS_E1, POS_E2, TENSOR].
    shls : torch.Tensor, shape (4,) int32
        Shell indices; shls[0]=i, shls[1]=j (k and l are unused for 1e).
    atm : torch.Tensor, shape (natm, ATM_SLOTS) int32
        Atom array in libcint layout.
    natm : int
        Number of atoms.
    bas : torch.Tensor, shape (nbas, BAS_SLOTS) int32
        Basis-shell array in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor, shape (nenv,) float64
        Environment array (coordinates, exponents, coefficients).

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    .. [2] S. Obara, A. Saika, "Efficient recursive computation of molecular
       integrals over Cartesian Gaussian functions." Journal of Chemical
       Physics, 84(7), 3963–3974 (1986).
    """
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
    """Compute the flat g-array index map for a one-electron shell pair.

    For every Cartesian component (i, j) of the shell pair, encodes the
    three (x, y, z) offsets into the g-array as a triplet, producing a
    flat index array of length nf * 3 that is used in the ``extract_gout``
    step.

    Parameters
    ----------
    envs : CINTEnvVars
        Populated environment struct (must have been initialised by
        :func:`init_envvars_1e`).

    Returns
    -------
    torch.Tensor, shape (nf*3,) int32
        Flat index map; triplet ``[3*n, 3*n+1, 3*n+2]`` gives the x, y, z
        g-array offsets for the n-th Cartesian component pair.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Pre-compute g-array index maps for all angular-momentum combinations.

    Iterates over all (l_i, l_j, ...) tuples up to the maximum angular
    momentum present in ``bas``, creates a fake one-shell-per-l basis,
    calls ``finit`` + ``findex_xyz`` for each combination, and stores the
    result in a flat list indexed by the base-LMAX1 encoded tuple.

    Parameters
    ----------
    finit : callable
        Environment initialiser (e.g. :func:`init_envvars_1e`).
    findex_xyz : callable
        Index-map builder (e.g. :func:`compute_g_index_1e`).
    order : int
        Number of shells in the integral (2 for 1e, 3 for 3c2e, 4 for 2e).
    max_l_override : int
        If > 0, caps the maximum l to min(max_l_override, max_l_in_basis).
    ng : torch.Tensor, shape (8,) int32
        Integral type descriptor.
    atm, bas, env : torch.Tensor
        Molecular arrays in libcint layout.
    natm, nbas : int
        Number of atoms and shells.

    Returns
    -------
    list
        Flat list of length ``(max_l+1) * LMAX1^(order-1)`` where each
        occupied entry is a ``(nf*3,) int32`` index tensor.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Build a CINTOpt for a one-electron integral type from an ng list.

    Computes log-max coefficients, non-zero contraction tables, and
    g-array index maps for the 1e integral specified by ``ng_list``.

    Parameters
    ----------
    ng_list : list of int
        8-element integral type descriptor [IINC, JINC, KINC, LINC,
        GSHIFT, POS_E1, POS_E2, TENSOR].
    atm, bas, env : torch.Tensor
        Molecular arrays in libcint layout.
    natm, nbas : int
        Number of atoms and shells.

    Returns
    -------
    CINTOpt
        Populated optimizer struct.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Build a CINTOpt for overlap integrals <i|j>.

    Parameters
    ----------
    opt_ref : object
        Unused placeholder (kept for API compatibility with libcint).
    atm, bas, env : torch.Tensor
        Molecular arrays in libcint layout.
    natm, nbas : int
        Number of atoms and shells.

    Returns
    -------
    CINTOpt
        Optimizer for overlap integral evaluation.

    References
    ----------
    .. [1] S. Obara, A. Saika, "Efficient recursive computation of molecular
       integrals over Cartesian Gaussian functions." Journal of Chemical
       Physics, 84(7), 3963–3974 (1986).
    """
    return build_optimizer_1e([0, 0, 0, 0, 0, 1, 1, 1], atm, natm, bas, nbas, env)


def build_kinetic_optimizer(opt_ref, atm, natm, bas, nbas, env):
    """Build a CINTOpt for kinetic energy integrals <i| -1/2 nabla^2 |j>.

    Parameters
    ----------
    opt_ref : object
        Unused placeholder (kept for API compatibility with libcint).
    atm, bas, env : torch.Tensor
        Molecular arrays in libcint layout.
    natm, nbas : int
        Number of atoms and shells.

    Returns
    -------
    CINTOpt
        Optimizer for kinetic energy integral evaluation.

    References
    ----------
    .. [1] M. Head-Gordon, J. A. Pople, "A method for two-electron Gaussian
       integral and integral derivative evaluation using recurrence relations."
       Journal of Chemical Physics, 89(9), 5777–5786 (1988).
    """
    return build_optimizer_1e([0, 2, 0, 0, 2, 1, 1, 1], atm, natm, bas, nbas, env)


def build_nuclear_optimizer(opt_ref, atm, natm, bas, nbas, env):
    """Build a CINTOpt for nuclear attraction integrals <i| sum_A Z_A/r_A |j>.

    Parameters
    ----------
    opt_ref : object
        Unused placeholder (kept for API compatibility with libcint).
    atm, bas, env : torch.Tensor
        Molecular arrays in libcint layout.
    natm, nbas : int
        Number of atoms and shells.

    Returns
    -------
    CINTOpt
        Optimizer for nuclear attraction integral evaluation.

    References
    ----------
    .. [1] S. Obara, A. Saika, "Efficient recursive computation of molecular
       integrals over Cartesian Gaussian functions." Journal of Chemical
       Physics, 84(7), 3963–3974 (1986).
    .. [2] M. Dupuis, J. Rys, H. F. King, "Evaluation of the molecular
       integrals over Gaussian basis functions." Journal of Chemical Physics,
       65(1), 111–116 (1976).
    """
    return build_optimizer_1e([0, 0, 0, 0, 0, 1, 0, 1], atm, natm, bas, nbas, env)


def init_envvars_2e(envs: CINTEnvVars, ng: torch.Tensor, shls: torch.Tensor,
                    atm: torch.Tensor, natm: int,
                    bas: torch.Tensor, nbas: int, env: torch.Tensor) -> None:
    """Populate a CINTEnvVars struct for a two-electron integral over a shell quartet.

    Sets angular momenta, Gaussian product centers, g-array strides, the
    number of Rys roots, and selects the 2D→4D recurrence variant
    (ibase/kbase logic) for the (i, j, k, l) shell quartet described by
    the integral type descriptor ``ng``.

    Parameters
    ----------
    envs : CINTEnvVars
        Output struct populated in-place.
    ng : torch.Tensor, shape (8,) int32
        Integral type descriptor.
    shls : torch.Tensor, shape (4,) int32
        Shell indices (i, j, k, l).
    atm, bas, env : torch.Tensor
        Molecular arrays in libcint layout.
    natm, nbas : int
        Number of atoms and shells.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    .. [2] S. Obara, A. Saika, "Efficient recursive computation of molecular
       integrals over Cartesian Gaussian functions." Journal of Chemical
       Physics, 84(7), 3963–3974 (1986).
    """
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
    """Compute the flat g-array index map for a two-electron shell quartet.

    Encodes the four Cartesian indices (i, j, k, l) into (x, y, z) g-array
    offsets, producing a ``(nf*3,) int32`` tensor analogous to
    :func:`compute_g_index_1e` but for four shells.

    Parameters
    ----------
    envs : CINTEnvVars
        Populated environment struct (must have been initialised by
        :func:`init_envvars_2e`).

    Returns
    -------
    torch.Tensor, shape (nf*3,) int32
        Flat g-array index map for the current shell quartet.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Compute Schwarz-screened PairData for all primitive pairs in a shell pair.

    For each (ip, jp) primitive pair computes the Gaussian product center
    ``rij``, the overlap prefactor ``eij = exp(-a_i a_j |R_i-R_j|^2 / a_ij)``,
    and the screening value ``cceij``.  Pairs with ``cceij >= expcutoff``
    are stored with zero weight; all others are flagged as significant.

    Parameters
    ----------
    ai, aj : torch.Tensor, shape (iprim,) / (jprim,) float64
        Primitive exponents for shells i and j.
    ri, rj : torch.Tensor, shape (3,) float64
        Atom positions for shells i and j.
    log_maxci, log_maxcj : torch.Tensor, shape (iprim,) / (jprim,) float64
        Log-max contraction coefficients from :func:`compute_log_max_coeff`.
    li_ceil, lj_ceil : int
        Effective angular momenta including derivative increments.
    iprim, jprim : int
        Number of primitives in shells i and j.
    rr_ij : float
        Squared inter-atomic distance |R_i - R_j|^2.
    expcutoff : float
        Screening threshold (default 60.0).

    Returns
    -------
    pairdata : list of PairData
        Length iprim * jprim, in (jp, ip) order.
    empty : bool
        True if all pairs were screened out.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
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
    """Compute Schwarz-screened shell-pair data for all (i, j) shell combinations.

    Iterates over all lower-triangular shell pairs (i, j) with j <= i and
    calls :func:`compute_pair_data` for each pair.  The resulting
    ``PairData`` objects are stored in a flat list indexed by
    ``i * nbas + j``.  If the primitive count exceeds
    ``MAX_PGTO_FOR_PAIRDATA``, ``None`` is returned and screening is
    disabled.  The transpose pair (j, i) is filled by copying and
    transposing the primitive index order.

    Parameters
    ----------
    ng : torch.Tensor
        1-D integer tensor of shape (8,) containing the libcint ``ng``
        control vector (angular increment flags, Rys root count hint, etc.).
    log_max_coeff : list of torch.Tensor
        Per-shell log-maximum contraction coefficients as returned by
        :func:`compute_log_max_coeffs`.
    atm : torch.Tensor
        Integer atom table of shape (natm, 6) in libcint layout.
    bas : torch.Tensor
        Integer basis table of shape (nbas, 8) in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor
        Floating-point environment array holding exponents, coefficients,
        coordinates, and control parameters.

    Returns
    -------
    list or None
        Flat list of length ``nbas * nbas`` whose entry at ``i * nbas + j``
        is either a list of :class:`PairData` objects (one per primitive
        pair), the sentinel ``NOVALUE`` (screened out), or ``None`` (not
        yet populated).  Returns ``None`` when screening is disabled
        because the primitive count exceeds ``MAX_PGTO_FOR_PAIRDATA``.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    .. [2] R. Lindh, U. Ryu, B. Liu, "The reduced multiplication scheme
       of the Rys quadrature and new recurrence relations for auxiliary
       function based two-electron integral evaluation." Journal of
       Chemical Physics, 95(8), 5889–5897 (1991).
    """
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
    """Build a fully populated 2-electron integral optimizer.

    Assembles a :class:`CINTOpt` object containing all data structures
    required for efficient screening and index look-up during
    4-center 2-electron repulsion integral (ERI) evaluation: per-shell
    log-maximum coefficients, Schwarz-screened shell-pair data, non-zero
    contraction coefficients, and the precomputed ``index_xyz`` tables.

    Parameters
    ----------
    ng : torch.Tensor
        1-D integer tensor of shape (8,) containing the libcint ``ng``
        control vector (IINC, JINC, KINC, LINC, POS_E1, POS_E2, TENSOR,
        GSHIFT flags).
    atm : torch.Tensor
        Integer atom table of shape (natm, 6) in libcint layout.
    natm : int
        Number of atoms.
    bas : torch.Tensor
        Integer basis table of shape (nbas, 8) in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor
        Floating-point environment array holding exponents, coefficients,
        coordinates, and control parameters.

    Returns
    -------
    CINTOpt
        Fully populated optimizer ready to be passed to
        :func:`primitive_loop_2e` / :func:`driver_2e_sph`.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    .. [2] M. Dupuis, J. Rys, H. F. King, "Evaluation of the molecular
       integrals over Gaussian basis functions." Journal of Chemical
       Physics, 65(1), 111–116 (1976).
    """
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
    """Populate a :class:`CINTEnvVars` instance for a 3-center 2-electron integral.

    Fills every field of ``envs`` that is needed by the Rys quadrature
    engine to evaluate the 3-center 2-electron integral
    (ij|k) = integral phi_i(r1) phi_j(r1) (1/r12) phi_k(r2) dr1 dr2
    over a shell triple (i, j, k).  Angular momenta, contraction counts,
    Cartesian sizes, recurrence strides, and the g-buffer size are all
    computed here.  The function also resolves the IB/JB base choice
    (``ibase``) and sets the ``f_g0_2d4d`` tag accordingly.

    Parameters
    ----------
    envs : CINTEnvVars
        Mutable environment struct to populate in-place.
    ng : torch.Tensor
        1-D integer tensor of shape (8,) (libcint ``ng`` control vector).
    shls : torch.Tensor
        1-D integer tensor ``[i_sh, j_sh, k_sh]`` giving the shell indices.
    atm : torch.Tensor
        Integer atom table of shape (natm, 6) in libcint layout.
    natm : int
        Number of atoms.
    bas : torch.Tensor
        Integer basis table of shape (nbas, 8) in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor
        Floating-point environment array.

    Returns
    -------
    None
        Modifies ``envs`` in place.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    .. [2] S. Obara and A. Saika, "Efficient recursive computation of
       molecular integrals over Cartesian Gaussian functions." Journal
       of Chemical Physics, 84(7), 3963–3974 (1986).
    """
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
    """Build an optimizer for 3-center 2-electron integrals (ij|k).

    Convenience entry point that constructs a :class:`CINTOpt` for
    3-center 2-electron integrals.  Uses a fixed ``ng`` vector
    ``[0,0,0,0,0,1,1,1]`` appropriate for standard ERI-like 3c2e
    kernels and delegates shell-pair precomputation to
    :func:`precompute_shell_pairs` and index generation to
    :func:`generate_index_xyz` via :func:`init_envvars_3c2e`.

    Parameters
    ----------
    opt_ref : object
        Unused reference placeholder kept for API compatibility with
        libcint's C interface (``CINTOpt **opt``).
    atm : torch.Tensor
        Integer atom table of shape (natm, 6) in libcint layout.
    natm : int
        Number of atoms.
    bas : torch.Tensor
        Integer basis table of shape (nbas, 8) in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor
        Floating-point environment array.

    Returns
    -------
    CINTOpt
        Optimizer containing log-max coefficients, shell-pair data,
        non-zero contraction info, and ``index_xyz`` tables for 3c2e.

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    .. [2] M. Dupuis, J. Rys, H. F. King, "Evaluation of the molecular
       integrals over Gaussian basis functions." Journal of Chemical
       Physics, 65(1), 111–116 (1976).
    """
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
    """Build an optimizer for 4-center 2-electron repulsion integrals (ij|kl).

    Convenience entry point that constructs a :class:`CINTOpt` for
    standard 4-center ERIs.  Uses a fixed ``ng`` vector
    ``[0,0,0,0,0,1,1,1]`` and delegates to :func:`build_optimizer_2e`.

    Parameters
    ----------
    opt_ref : object
        Unused reference placeholder kept for API compatibility with
        libcint's C interface (``CINTOpt **opt``).
    atm : torch.Tensor
        Integer atom table of shape (natm, 6) in libcint layout.
    natm : int
        Number of atoms.
    bas : torch.Tensor
        Integer basis table of shape (nbas, 8) in libcint layout.
    nbas : int
        Number of basis shells.
    env : torch.Tensor
        Floating-point environment array.

    Returns
    -------
    CINTOpt
        Optimizer containing log-max coefficients, shell-pair data,
        non-zero contraction info, and ``index_xyz`` tables for 4-center
        2-electron integrals.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.analytical_integrators_torch.optimizer import (
    ...     build_2e_optimizer, make_fake_basis)
    >>> mol = None  # doctest: +SKIP
    >>> opt = build_2e_optimizer(None, atm, natm, bas, nbas, env)  # doctest: +SKIP

    References
    ----------
    .. [1] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    .. [2] M. Dupuis, J. Rys, H. F. King, "Evaluation of the molecular
       integrals over Gaussian basis functions." Journal of Chemical
       Physics, 65(1), 111–116 (1976).
    """
    ng = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
    return build_optimizer_2e(ng, atm, natm, bas, nbas, env)
