"""
Pure Python/NumPy implementations of libcint integral functions.

Replaces the C libcint functions:
  - int1e_ovlp_sph  (overlap)
  - int1e_kin_sph   (kinetic energy)
  - int1e_nuc_sph   (nuclear attraction)
  - int2e_ar12b_sph (electron repulsion)

And the GTO driver functions:
  - GTOint2c        (2-center integral matrix assembly)
  - GTOnr2e_fill_s1 / GTOnr2e_fill_drv (4-center integral assembly)
"""
import math
import numpy as np
from deepchem.utils.analytical_integrators.optimizer import (
    CINTEnvVars, CINTcart_comp, CINTcommon_fac_sp,
    CINTinit_int1e_EnvVars, CINTg1e_index_xyz,
    CINTinit_int2e_EnvVars, CINTg2e_index_xyz,
    CINTinit_int3c2e_EnvVars,
    ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF,
    PTR_COORD, EXPCUTOFF, MIN_EXPCUTOFF, PTR_EXPCUTOFF,
    BAS_SLOTS, ATM_SLOTS,
)
from deepchem.utils.analytical_integrators.spherical import rys_roots

SQRTPI = math.sqrt(math.pi)

# ================================================================
# Cartesian-to-spherical transformation matrices
# ================================================================
# Standard real solid harmonic transformation coefficients.
# For l=0,1: cart and sph are equivalent (CINTcommon_fac_sp handles norm).
# For l>=2: full transformation matrices needed.

def _cart2sph_matrix(l):
    """
    Return the (2l+1, nf_cart) transformation matrix from Cartesian
    to real solid harmonics for angular momentum l.
    Uses the same convention as libcint's c2s_bra/ket tables.
    """
    if l == 0:
        return np.array([[1.0]])
    elif l == 1:
        # libcint order: cart = (x, y, z), sph = (y, z, x) i.e. m=(-1,0,+1)
        return np.array([
            [0.0, 1.0, 0.0],  # Y_{1,-1} ~ y
            [0.0, 0.0, 1.0],  # Y_{1,0}  ~ z
            [1.0, 0.0, 0.0],  # Y_{1,+1} ~ x
        ])
    elif l == 2:
        # Cart order: xx, xy, xz, yy, yz, zz (libcint canonical)
        # Sph order: m = -2, -1, 0, +1, +2
        s3 = math.sqrt(3.0)
        return np.array([
            #  xx        xy       xz       yy       yz       zz
            [0.0,      s3,       0.0,     0.0,     0.0,     0.0     ],  # d_{-2} = sqrt(3)*xy
            [0.0,      0.0,      0.0,     0.0,     s3,      0.0     ],  # d_{-1} = sqrt(3)*yz
            [-0.5,     0.0,      0.0,    -0.5,     0.0,     1.0     ],  # d_0   = zz - (xx+yy)/2
            [0.0,      0.0,      s3,      0.0,     0.0,     0.0     ],  # d_{+1} = sqrt(3)*xz
            [s3/2.0,   0.0,      0.0,    -s3/2.0,  0.0,     0.0     ],  # d_{+2} = sqrt(3)/2*(xx-yy)
        ])
    elif l == 3:
        # Cart order (libcint canonical, decreasing nx then ny):
        # xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
        s = math.sqrt
        # f-orbital transformation
        return np.array([
            # xxx       xxy      xxz      xyy      xyz      xzz      yyy      yyz      yzz      zzz
            [0,         s(10)/4, 0,       0,       0,       0,       -s(10)/4*3,0,       0,       0        ],  # f_{-3}
            [0,         0,       0,       0,       s(15),   0,       0,        0,        0,       0        ],  # f_{-2}
            [0,         -s(6)/4, 0,       0,       0,       0,       -s(6)/4,  0,        s(6),    0        ],  # f_{-1} -- FIXME check
            [0,         0,       -3./2.,  0,       0,       0,       0,        -3./2.,   0,       1.       ],  # f_0    -- FIXME check
            [-s(6)/4,   0,       0,       -s(6)/4, 0,       s(6)/2,  0,        0,        0,       0        ],  # f_{+1} -- FIXME check
            [0,         0,       s(15)/2, 0,       0,       0,       0,        -s(15)/2, 0,       0        ],  # f_{+2}
            [s(10)/4*3, 0,       0,       -s(10)/4,0,       0,       0,        0,        0,       0        ],  # f_{+3} -- FIXME check
        ])
    else:
        raise NotImplementedError(f"cart2sph not implemented for l={l}")


# Precompute and cache transformation matrices
_CART2SPH_CACHE = {}

def cart2sph_matrix(l):
    """Get cached cart2sph transformation matrix for angular momentum l."""
    if l not in _CART2SPH_CACHE:
        _CART2SPH_CACHE[l] = _cart2sph_matrix(l)
    return _CART2SPH_CACHE[l]


def c2s_sph_1e(gctr, i_l, j_l, i_ctr, j_ctr, nfi, nfj, nf):
    """
    Cartesian-to-spherical transformation for 1e integrals.

    Parameters
    ----------
    gctr : ndarray, shape (j_ctr * i_ctr * nf,)
        Contracted Cartesian integrals, laid out as blocks of nf per (ic, jc) pair.
    i_l, j_l : int
        Angular momenta of bra and ket shells.
    i_ctr, j_ctr : int
        Number of contractions for bra and ket.
    nfi, nfj : int
        Number of Cartesian components for i and j shells.
    nf : int
        nfi * nfj

    Returns
    -------
    out : ndarray, shape (di * i_ctr, dj * j_ctr) where di=2*i_l+1, dj=2*j_l+1
    """
    di = 2 * i_l + 1
    dj = 2 * j_l + 1

    c2s_i = cart2sph_matrix(i_l)  # (di, nfi)
    c2s_j = cart2sph_matrix(j_l)  # (dj, nfj)

    out = np.zeros((di * i_ctr, dj * j_ctr))

    for jc in range(j_ctr):
        for ic in range(i_ctr):
            # Extract Cartesian block for this contraction pair
            offset = (jc * i_ctr + ic) * nf
            cart_block = gctr[offset:offset + nf].reshape(nfj, nfi)
            # Transform: sph = c2s_i @ cart_block.T @ c2s_j.T  →  (di, dj)
            sph_block = c2s_i @ cart_block.T @ c2s_j.T
            # Place into output
            i0 = ic * di
            j0 = jc * dj
            out[i0:i0 + di, j0:j0 + dj] = sph_block

    return out


def c2s_sph_2e1(gctr, i_l, j_l, k_l, l_l, x_ctr, nfi, nfj, nfk, nfl, nf):
    """
    Cartesian-to-spherical transformation for 2e integrals.

    Returns
    -------
    out : ndarray, shape (di*i_ctr, dj*j_ctr, dk*k_ctr, dl*l_ctr)
    """
    i_ctr, j_ctr, k_ctr, l_ctr = x_ctr
    di = 2 * i_l + 1
    dj = 2 * j_l + 1
    dk = 2 * k_l + 1
    dl = 2 * l_l + 1

    c2s_i = cart2sph_matrix(i_l)
    c2s_j = cart2sph_matrix(j_l)
    c2s_k = cart2sph_matrix(k_l)
    c2s_l = cart2sph_matrix(l_l)

    out = np.zeros((di * i_ctr, dj * j_ctr, dk * k_ctr, dl * l_ctr))

    for lc in range(l_ctr):
        for kc in range(k_ctr):
            for jc in range(j_ctr):
                for ic in range(i_ctr):
                    idx = ((lc * k_ctr + kc) * j_ctr + jc) * i_ctr + ic
                    offset = idx * nf
                    cart_4d = gctr[offset:offset + nf].reshape(nfj, nfl, nfk, nfi)
                    tmp = np.tensordot(c2s_i, cart_4d, axes=([1], [3]))  # (di, nfj, nfl, nfk)
                    tmp = np.tensordot(c2s_j, tmp, axes=([1], [1]))      # (dj, di, nfl, nfk)
                    tmp = np.tensordot(c2s_k, tmp, axes=([1], [3]))      # (dk, dj, di, nfl)
                    tmp = np.tensordot(c2s_l, tmp, axes=([1], [3]))      # (dl, dk, dj, di)
                    i0, j0, k0, l0 = ic * di, jc * dj, kc * dk, lc * dl
                    out[i0:i0+di, j0:j0+dj, k0:k0+dk, l0:l0+dl] = tmp.transpose(3, 2, 1, 0)

    return out


# ================================================================
# G-value generation (recurrence relations)
# ================================================================

def _CINTg_vrr_hrr(g, envs, gz0_fac, rir0, cfac, aij):
    """Unified vertical + horizontal recurrence for 1e g-values.
    gz0_fac: initial gz[0] value.  rir0: shift vector for vertical recurrence.
    cfac: prefactor for vertical coeff (0.5/aij for ovlp, 0.5*(1-t2)/aij for nuc).
    """
    nmax = envs.li_ceil + envs.lj_ceil
    lj = envs.lj_ceil
    dj = envs.g_stride_j
    g_size = envs.g_size
    gx, gy, gz = g[:g_size], g[g_size:2 * g_size], g[2 * g_size:3 * g_size]
    rirj = envs.ri - envs.rj

    gx[0] = 1.0;  gy[0] = 1.0;  gz[0] = gz0_fac
    if nmax > 0:
        gx[1] = -rir0[0] * gx[0]
        gy[1] = -rir0[1] * gy[0]
        gz[1] = -rir0[2] * gz[0]
    for i in range(1, nmax):
        c = cfac * i
        gx[i + 1] = c * gx[i - 1] - rir0[0] * gx[i]
        gy[i + 1] = c * gy[i - 1] - rir0[1] * gy[i]
        gz[i + 1] = c * gz[i - 1] - rir0[2] * gz[i]
    for j in range(1, lj + 1):
        ptr = dj * j
        for i in range(ptr, ptr + nmax - j + 1):
            gx[i] = gx[i + 1 - dj] + rirj[0] * gx[i - dj]
            gy[i] = gy[i + 1 - dj] + rirj[1] * gy[i - dj]
            gz[i] = gz[i + 1 - dj] + rirj[2] * gz[i - dj]


def CINTg_ovlp(g, ai, aj, fac, envs):
    """Generate g-values for overlap integrals."""
    aij = ai + aj
    rir0 = envs.ri - (ai * envs.ri + aj * envs.rj) / aij
    _CINTg_vrr_hrr(g, envs, SQRTPI * math.pi * fac, rir0, 0.5 / aij, aij)


def CINTg_nuc(g, aij, rij, cr, t2, fac, envs):
    """Generate g-values for nuclear attraction integrals."""
    rir0 = envs.ri - (rij + t2 * (cr - rij))
    _CINTg_vrr_hrr(g, envs, 2.0 * math.pi * fac, rir0, 0.5 * (1.0 - t2) / aij, aij)


# ================================================================
# Nabla (derivative) operators for kinetic energy
# ================================================================

def CINTnabla1j_1e(f, g, li, lj, lk, envs):
    """Compute nabla_j g-values for kinetic energy integrals."""
    dj = envs.g_stride_j
    aj2 = -2.0 * envs.aj
    g_size = envs.g_size
    gx, gy, gz = g[:g_size], g[g_size:2*g_size], g[2*g_size:3*g_size]
    fx, fy, fz = f[:g_size], f[g_size:2*g_size], f[2*g_size:3*g_size]

    for k in range(lk + 1):
        dk_off = 0  # no dk for 1e
        ptr = dk_off
        # j=0: f = -2*aj*g[...,1,...]
        for i in range(ptr, ptr + li + 1):
            fx[i] = aj2 * gx[i + dj]
            fy[i] = aj2 * gy[i + dj]
            fz[i] = aj2 * gz[i + dj]
        # j>=1: f = j*g[...,j-1,...] - 2*aj*g[...,j+1,...]
        for j in range(1, lj + 1):
            ptr = dj * j + dk_off
            for i in range(ptr, ptr + li + 1):
                fx[i] = j * gx[i - dj] + aj2 * gx[i + dj]
                fy[i] = j * gy[i - dj] + aj2 * gy[i + dj]
                fz[i] = j * gz[i - dj] + aj2 * gz[i + dj]


# ================================================================
# Gout extraction functions
# ================================================================

def gout_1e_ovlp(gout, g, idx, envs):
    """Extract overlap gout using vectorized indexing."""
    ix = idx[0::3]
    iy = idx[1::3]
    iz = idx[2::3]
    gout[:envs.nf] += g[ix] * g[iy] * g[iz]


def gout_1e_nuc(gout, g, idx, envs):
    """Extract nuclear attraction gout (same as overlap)."""
    gout_1e_ovlp(gout, g, idx, envs)


def gout_1e_kin(gout, g, idx, envs):
    """Extract kinetic energy gout: <i| -1/2 nabla^2 |j>."""
    g_size = envs.g_size
    g_len = g_size * 3
    g0 = g[:g_len]
    g1 = g[g_len:2 * g_len]
    g2 = g[2 * g_len:3 * g_len]
    g3 = g[3 * g_len:4 * g_len]

    CINTnabla1j_1e(g1, g0, envs.i_l, envs.j_l, 0, envs)
    CINTnabla1j_1e(g2, g0, envs.i_l, envs.j_l + 1, 0, envs)
    CINTnabla1j_1e(g3, g2, envs.i_l, envs.j_l, 0, envs)

    ix = idx[0::3]
    iy = idx[1::3]
    iz = idx[2::3]
    gout[:envs.nf] += -(g3[ix] * g0[iy] * g0[iz] +
                         g0[ix] * g3[iy] * g0[iz] +
                         g0[ix] * g0[iy] * g3[iz])


def gout_2e_ar12b(gout, g, idx, envs, gout_empty):
    """Extract 2e gout: sum over Rys roots of g[ix]*g[iy]*g[iz]."""
    nroots = envs.nrys_roots
    ix = idx[0::3]
    iy = idx[1::3]
    iz = idx[2::3]
    s = np.zeros(envs.nf)
    for i in range(nroots):
        s += g[ix + i] * g[iy + i] * g[iz + i]
    if gout_empty:
        gout[:envs.nf] = s
    else:
        gout[:envs.nf] += s


# ================================================================
# Primitive-to-contracted transformation
# ================================================================

# ================================================================
# 1e integral loops
# ================================================================

def CINT1e_loop(envs, atm, bas, env):
    """
    1e overlap/kinetic primitive loop.
    Returns (gctr, has_value) where gctr is the contracted Cartesian integrals.
    """
    shls = envs.shls
    i_sh, j_sh = int(shls[0]), int(shls[1])
    i_l, j_l = envs.i_l, envs.j_l
    i_ctr, j_ctr = int(envs.x_ctr[0]), int(envs.x_ctr[1])
    i_prim = int(bas[i_sh, NPRIM_OF])
    j_prim = int(bas[j_sh, NPRIM_OF])
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor
    ri, rj = envs.ri, envs.rj
    ai_arr = env[bas[i_sh, PTR_EXP]:bas[i_sh, PTR_EXP] + i_prim]
    aj_arr = env[bas[j_sh, PTR_EXP]:bas[j_sh, PTR_EXP] + j_prim]
    ci = env[bas[i_sh, PTR_COEFF]:bas[i_sh, PTR_COEFF] + i_prim * i_ctr]
    cj = env[bas[j_sh, PTR_COEFF]:bas[j_sh, PTR_COEFF] + j_prim * j_ctr]

    idx = CINTg1e_index_xyz(envs)
    rrij = float(np.sum((ri - rj) ** 2))
    fac = envs.common_factor * CINTcommon_fac_sp(i_l) * CINTcommon_fac_sp(j_l)
    expcutoff = envs.expcutoff

    nc = nf * i_ctr * j_ctr
    gctr = np.zeros(nc * n_comp)
    has_value = False

    # Number of g-array sections needed
    # For kinetic energy, gout uses 4 sections (g0,g1,g2,g3), each g_size*3.
    # The C code allocates (1<<gbits)+1 sections but relies on stack overflow
    # into gout/gctri space. We allocate extra to be safe.
    gbits = envs.gbits
    g_sections = max((1 << gbits) + 1, 4)  # at least 4 for kinetic
    g_alloc = envs.g_size * 3 * g_sections

    for jp in range(j_prim):
        envs.aj = aj_arr[jp]
        gctri = np.zeros(nf * i_ctr * n_comp)

        for ip in range(i_prim):
            envs.ai = ai_arr[ip]
            aij = ai_arr[ip] + aj_arr[jp]
            eij = (ai_arr[ip] * aj_arr[jp] / aij) * rrij
            if eij > expcutoff:
                continue
            has_value = True

            dij = math.exp(-eij) / (aij * math.sqrt(aij)) * fac

            g = np.zeros(g_alloc)
            CINTg_ovlp(g, ai_arr[ip], aj_arr[jp], dij, envs)

            gout = np.zeros(nf * n_comp)
            envs.f_gout(gout, g, idx, envs)

            # Contract over i primitives
            for n in range(n_comp):
                block = gout[n * nf:(n + 1) * nf]
                for ic in range(i_ctr):
                    c = ci[i_prim * ic + ip]
                    if c != 0:
                        offset = n * nf * i_ctr + ic * nf
                        gctri[offset:offset + nf] += c * block

        # Contract over j primitives
        for n in range(n_comp):
            for jc in range(j_ctr):
                c = cj[j_prim * jc + jp]
                if c != 0:
                    for ic in range(i_ctr):
                        src_off = n * nf * i_ctr + ic * nf
                        dst_off = n * nf * i_ctr * j_ctr + (jc * i_ctr + ic) * nf
                        gctr[dst_off:dst_off + nf] += c * gctri[src_off:src_off + nf]

    return gctr, has_value


def CINT1e_nuc_loop(envs, atm, bas, env, charge_fac, nuc_id):
    """
    1e nuclear attraction primitive loop.
    Returns (gctr, has_value).
    """
    shls = envs.shls
    i_sh, j_sh = int(shls[0]), int(shls[1])
    i_l, j_l = envs.i_l, envs.j_l
    i_ctr, j_ctr = int(envs.x_ctr[0]), int(envs.x_ctr[1])
    i_prim = int(bas[i_sh, NPRIM_OF])
    j_prim = int(bas[j_sh, NPRIM_OF])
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor
    ri, rj = envs.ri, envs.rj
    ai_arr = env[bas[i_sh, PTR_EXP]:bas[i_sh, PTR_EXP] + i_prim]
    aj_arr = env[bas[j_sh, PTR_EXP]:bas[j_sh, PTR_EXP] + j_prim]
    ci = env[bas[i_sh, PTR_COEFF]:bas[i_sh, PTR_COEFF] + i_prim * i_ctr]
    cj = env[bas[j_sh, PTR_COEFF]:bas[j_sh, PTR_COEFF] + j_prim * j_ctr]

    idx = CINTg1e_index_xyz(envs)

    # Nuclear position
    if nuc_id < 0:
        PTR_RINV_ORIG = 4
        cr = env[PTR_RINV_ORIG:PTR_RINV_ORIG + 3]
    else:
        cr = env[atm[nuc_id, PTR_COORD]:atm[nuc_id, PTR_COORD] + 3]

    rrij = float(np.sum((ri - rj) ** 2))
    fac = charge_fac * envs.common_factor * CINTcommon_fac_sp(i_l) * CINTcommon_fac_sp(j_l)
    expcutoff = envs.expcutoff

    nc = nf * i_ctr * j_ctr
    gctr = np.zeros(nc * n_comp)
    has_value = False

    g_alloc = envs.g_size * 3

    for jp in range(j_prim):
        envs.aj = aj_arr[jp]
        gctri = np.zeros(nf * i_ctr * n_comp)

        for ip in range(i_prim):
            envs.ai = ai_arr[ip]
            aij = ai_arr[ip] + aj_arr[jp]
            eij = (ai_arr[ip] * aj_arr[jp] / aij) * rrij
            if eij > expcutoff:
                continue
            has_value = True

            rij = (ai_arr[ip] * ri + aj_arr[jp] * rj) / aij
            tau = 1.0  # point charge model (no Gaussian nuclear model)
            x = aij * float(np.sum((rij - cr) ** 2)) * tau * tau
            nroots = envs.nrys_roots
            u, w = rys_roots(nroots, x)

            dij = math.exp(-eij) / aij * fac

            gout = np.zeros(nf * n_comp)
            for iroot in range(nroots):
                t2 = u[iroot] / (1.0 + u[iroot]) * tau * tau
                g = np.zeros(g_alloc)
                CINTg_nuc(g, aij, rij, cr, t2, dij * w[iroot] * tau, envs)
                envs.f_gout(gout, g, idx, envs)

            # Contract over i primitives
            for n in range(n_comp):
                block = gout[n * nf:(n + 1) * nf]
                for ic in range(i_ctr):
                    c = ci[i_prim * ic + ip]
                    if c != 0:
                        offset = n * nf * i_ctr + ic * nf
                        gctri[offset:offset + nf] += c * block

        # Contract over j primitives
        for n in range(n_comp):
            for jc in range(j_ctr):
                c = cj[j_prim * jc + jp]
                if c != 0:
                    for ic in range(i_ctr):
                        src_off = n * nf * i_ctr + ic * nf
                        dst_off = n * nf * i_ctr * j_ctr + (jc * i_ctr + ic) * nf
                        gctr[dst_off:dst_off + nf] += c * gctri[src_off:src_off + nf]

    return gctr, has_value


# ================================================================
# 1e integral driver
# ================================================================

INT1E_TYPE_OVLP = 0
INT1E_TYPE_RINV = 1
INT1E_TYPE_NUC = 2


def CINT1e_drv(envs, atm, bas, env, int1e_type):
    """
    1e integral driver. Returns contracted spherical integrals as
    ndarray of shape (di*i_ctr, dj*j_ctr).
    """
    i_ctr, j_ctr = int(envs.x_ctr[0]), int(envs.x_ctr[1])
    nf = envs.nf
    nfi, nfj = envs.nfi, envs.nfj
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor
    nc = nf * i_ctr * j_ctr

    gctr = np.zeros(nc * n_comp)

    if int1e_type == INT1E_TYPE_OVLP:
        gctr, has_value = CINT1e_loop(envs, atm, bas, env)
    elif int1e_type == INT1E_TYPE_NUC:
        # Sum over all nuclei
        natm = envs.natm
        has_value = False
        for n in range(natm):
            charge = abs(int(atm[n, 0]))  # CHARGE_OF = column 0
            if charge != 0:
                gc_n, hv = CINT1e_nuc_loop(envs, atm, bas, env, -charge, n)
                gctr += gc_n
                has_value = has_value or hv
    elif int1e_type == INT1E_TYPE_RINV:
        gctr, has_value = CINT1e_nuc_loop(envs, atm, bas, env, 1.0, -1)

    # Cart-to-spherical transform
    return c2s_sph_1e(gctr, envs.i_l, envs.j_l, i_ctr, j_ctr, nfi, nfj, nf)


# ================================================================
# Top-level 1e integral functions
# ================================================================

def _int1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env, ng, f_gout, int1e_type, fac=1.0):
    """Common driver for all 1e spherical integrals."""
    envs = CINTEnvVars()
    CINTinit_int1e_EnvVars(envs, ng, shls, atm, natm, bas, nbas, env)
    envs.f_gout = f_gout
    envs.common_factor *= fac
    result = CINT1e_drv(envs, atm, bas, env, int1e_type)
    di = 2 * envs.i_l + 1
    dj = 2 * envs.j_l + 1
    i_ctr = int(envs.x_ctr[0])
    j_ctr = int(envs.x_ctr[1])
    naoi = dims[0]
    for jc in range(j_ctr):
        for ic in range(i_ctr):
            for j in range(dj):
                for i in range(di):
                    out[(jc * dj + j) * naoi + (ic * di + i)] = \
                        result[ic * di + i, jc * dj + j]
    return 1


def int1e_ovlp_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure Python int1e_ovlp_sph."""
    ng = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    return _int1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env,
                             ng, gout_1e_ovlp, INT1E_TYPE_OVLP)


def int1e_kin_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure Python int1e_kin_sph."""
    ng = np.array([0, 2, 0, 0, 2, 1, 1, 1], dtype=np.int32)
    return _int1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env,
                             ng, gout_1e_kin, INT1E_TYPE_OVLP, fac=0.5)


def int1e_nuc_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure Python int1e_nuc_sph."""
    ng = np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int32)
    return _int1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env,
                             ng, gout_1e_nuc, INT1E_TYPE_NUC)


# ================================================================
# 2e integral: g-value generation
# ================================================================

def CINTg0_2e_2d(g, bc, envs):
    """2D Rys polynomial recurrence for 2e integrals."""
    nroots = envs.nrys_roots
    nmax = envs.li_ceil + envs.lj_ceil
    mmax = envs.lk_ceil + envs.ll_ceil
    dm = envs.g2d_klmax
    dn = envs.g2d_ijmax
    g_size = envs.g_size

    gx = g[:g_size]
    gy = g[g_size:2 * g_size]
    gz = g[2 * g_size:3 * g_size]

    c00 = bc['c00']  # (nroots, 3)
    c0p = bc['c0p']  # (nroots, 3)
    b00 = bc['b00']  # (nroots,)
    b10 = bc['b10']  # (nroots,)
    b01 = bc['b01']  # (nroots,)

    for i in range(nroots):
        gx[i] = 1.0
        gy[i] = 1.0
        # gz[i] already set to w[i] * fac1

    # n-recurrence (ij direction)
    if nmax > 0:
        for i in range(nroots):
            gx[dn + i] = c00[i, 0] * gx[i]
            gy[dn + i] = c00[i, 1] * gy[i]
            gz[dn + i] = c00[i, 2] * gz[i]
        for n in range(1, nmax):
            off = n * dn
            for i in range(nroots):
                j = off + i
                gx[dn + j] = c00[i, 0] * gx[j] + n * b10[i] * gx[j - dn]
                gy[dn + j] = c00[i, 1] * gy[j] + n * b10[i] * gy[j - dn]
                gz[dn + j] = c00[i, 2] * gz[j] + n * b10[i] * gz[j - dn]

    # m-recurrence (kl direction)
    if mmax > 0:
        for i in range(nroots):
            gx[dm + i] = c0p[i, 0] * gx[i]
            gy[dm + i] = c0p[i, 1] * gy[i]
            gz[dm + i] = c0p[i, 2] * gz[i]
        for m in range(1, mmax):
            off = m * dm
            for i in range(nroots):
                j = off + i
                gx[dm + j] = c0p[i, 0] * gx[j] + m * b01[i] * gx[j - dm]
                gy[dm + j] = c0p[i, 1] * gy[j] + m * b01[i] * gy[j - dm]
                gz[dm + j] = c0p[i, 2] * gz[j] + m * b01[i] * gz[j - dm]

    # Cross terms (both nmax > 0 and mmax > 0)
    if nmax > 0 and mmax > 0:
        # (m=1..mmax, n=0): gx(1,m) = c0p*gx(0,m) + b00*gx(0,m-1) [at n=0+dn]
        # First: g(irys,1,1) = c0p*g(irys,0,1) + b00*g(irys,0,0)
        for i in range(nroots):
            gx[dn + i + dm] = c0p[i, 0] * gx[dn + i] + b00[i] * gx[i]
            gy[dn + i + dm] = c0p[i, 1] * gy[dn + i] + b00[i] * gy[i]
            gz[dn + i + dm] = c0p[i, 2] * gz[dn + i] + b00[i] * gz[i]

        for m in range(1, mmax):
            off = m * dm + dn
            for i in range(nroots):
                j = off + i
                gx[j + dm] = c0p[i, 0] * gx[j] + m * b01[i] * gx[j - dm] + b00[i] * gx[j - dn]
                gy[j + dm] = c0p[i, 1] * gy[j] + m * b01[i] * gy[j - dm] + b00[i] * gy[j - dn]
                gz[j + dm] = c0p[i, 2] * gz[j] + m * b01[i] * gz[j - dm] + b00[i] * gz[j - dn]

        # g(irys,m,n+1) = c00*g(irys,m,n) + n*b10*g(irys,m,n-1) + m*b00*g(irys,m-1,n)
        for m in range(1, mmax + 1):
            for n in range(1, nmax):
                off = m * dm + n * dn
                for i in range(nroots):
                    j = off + i
                    gx[j + dn] = c00[i, 0] * gx[j] + n * b10[i] * gx[j - dn] + m * b00[i] * gx[j - dm]
                    gy[j + dn] = c00[i, 1] * gy[j] + n * b10[i] * gy[j - dn] + m * b00[i] * gy[j - dm]
                    gz[j + dn] = c00[i, 2] * gz[j] + n * b10[i] * gz[j - dn] + m * b00[i] * gz[j - dm]


def _CINTg0_hrr_phase(gx, gy, gz, r, l_tgt, n_max, d_tgt, d_src, d_oth, l_oth, stride):
    """One phase of horizontal recurrence for 4D g-values.
    g[a] = r * g[a - d_tgt] + g[a - d_tgt + d_src]
    Loops: a in 1..l_tgt, b in 0..n_max-a, c in 0..l_oth.
    """
    for a in range(1, l_tgt + 1):
        for b in range(n_max - a + 1):
            for c in range(l_oth + 1):
                ptr = a * d_tgt + b * d_src + c * d_oth
                for n in range(ptr, ptr + stride):
                    gx[n] = r[0] * gx[n - d_tgt] + gx[n - d_tgt + d_src]
                    gy[n] = r[1] * gy[n - d_tgt] + gy[n - d_tgt + d_src]
                    gz[n] = r[2] * gz[n - d_tgt] + gz[n - d_tgt + d_src]


def _CINTg0_2d4d(g, envs, ij_args, kl_args):
    """Unified 4D horizontal recurrence with two phases.
    Each args tuple: (l_tgt, n_max, d_tgt, d_src, d_oth, l_oth, stride, r)
    """
    g_size = envs.g_size
    gx, gy, gz = g[:g_size], g[g_size:2 * g_size], g[2 * g_size:3 * g_size]
    for l_tgt, n_max, d_tgt, d_src, d_oth, l_oth, stride, r in (ij_args, kl_args):
        _CINTg0_hrr_phase(gx, gy, gz, r, l_tgt, n_max, d_tgt, d_src, d_oth, l_oth, stride)



def CINTg0_2e(g, fac, envs):
    """
    Compute 2e g-values: Rys roots/weights, then 2D recurrence, then 4D recurrence.
    Returns True if successful.
    """
    aij = envs.aij
    akl = envs.akl
    g_size = envs.g_size

    rijrkl = envs.rij - envs.rkl
    a1 = aij * akl
    a0 = a1 / (aij + akl)

    x = a0 * float(np.sum(rijrkl ** 2))
    nroots = envs.nrys_roots

    u, w = rys_roots(nroots, x)

    fac1 = math.sqrt(a0 / (a1 * a1 * a1)) * fac

    gz = g[2 * g_size:3 * g_size]

    if g_size == 1:
        g[0] = 1.0
        g[g_size] = 1.0
        gz[0] = w[0] * fac1
        return True

    rijrx = envs.rijrx
    rklrx = envs.rklrx

    c00 = np.zeros((nroots, 3))
    c0p = np.zeros((nroots, 3))
    b00 = np.zeros(nroots)
    b10 = np.zeros(nroots)
    b01 = np.zeros(nroots)

    for irys in range(nroots):
        u2 = a0 * u[irys]
        tmp4 = 0.5 / (u2 * (aij + akl) + a1)
        tmp5 = u2 * tmp4
        tmp1 = 2.0 * tmp5
        tmp2 = tmp1 * akl
        tmp3 = tmp1 * aij

        b00[irys] = tmp5
        b10[irys] = tmp5 + tmp4 * akl
        b01[irys] = tmp5 + tmp4 * aij

        c00[irys, 0] = rijrx[0] - tmp2 * rijrkl[0]
        c00[irys, 1] = rijrx[1] - tmp2 * rijrkl[1]
        c00[irys, 2] = rijrx[2] - tmp2 * rijrkl[2]
        c0p[irys, 0] = rklrx[0] + tmp3 * rijrkl[0]
        c0p[irys, 1] = rklrx[1] + tmp3 * rijrkl[1]
        c0p[irys, 2] = rklrx[2] + tmp3 * rijrkl[2]

        gz[irys] = w[irys] * fac1

    bc = {'c00': c00, 'c0p': c0p, 'b00': b00, 'b10': b10, 'b01': b01}

    CINTg0_2e_2d(g, bc, envs)
    # 4D recurrence via dict lookup
    nmax = envs.li_ceil + envs.lj_ceil
    mmax = envs.lk_ceil + envs.ll_ceil
    di, dk, dl, dj = envs.g_stride_i, envs.g_stride_k, envs.g_stride_l, envs.g_stride_j
    nr, rirj, rkrl = envs.nrys_roots, envs.rirj, envs.rkrl
    li, lj, lk, ll = envs.li_ceil, envs.lj_ceil, envs.lk_ceil, envs.ll_ceil
    _4d_args = {
        'lj2d4d': ((li, nmax, di, dj, dl, mmax, nr, rirj), (lk, mmax, dk, dl, dj, lj, dk, rkrl)),
        'kj2d4d': ((li, nmax, di, dj, dk, mmax, nr, rirj), (ll, mmax, dl, dk, dj, lj, dl, rkrl)),
        'ik2d4d': ((lj, nmax, dj, di, dl, mmax, nr, rirj), (ll, mmax, dl, dk, di, li, dl, rkrl)),
        'il2d4d': ((lj, nmax, dj, di, dk, mmax, nr, rirj), (lk, mmax, dk, dl, di, li, dk, rkrl)),
    }
    ij_args, kl_args = _4d_args[envs.f_g0_2d4d]
    _CINTg0_2d4d(g, envs, ij_args, kl_args)

    return True


# ================================================================
# 2e integral loop
# ================================================================

def CINT2e_loop_nopt(envs, atm, bas, env):
    """
    2e integral primitive loop (no optimizer).
    Returns (gctr, has_value).
    """
    shls = envs.shls
    i_sh, j_sh = int(shls[0]), int(shls[1])
    k_sh, l_sh = int(shls[2]), int(shls[3])
    i_ctr = int(envs.x_ctr[0])
    j_ctr = int(envs.x_ctr[1])
    k_ctr = int(envs.x_ctr[2])
    l_ctr = int(envs.x_ctr[3])
    i_prim = int(bas[i_sh, NPRIM_OF])
    j_prim = int(bas[j_sh, NPRIM_OF])
    k_prim = int(bas[k_sh, NPRIM_OF])
    l_prim = int(bas[l_sh, NPRIM_OF])

    ri, rj = envs.ri, envs.rj
    rk, rl = envs.rk, envs.rl
    ai_arr = env[bas[i_sh, PTR_EXP]:bas[i_sh, PTR_EXP] + i_prim]
    aj_arr = env[bas[j_sh, PTR_EXP]:bas[j_sh, PTR_EXP] + j_prim]
    ak_arr = env[bas[k_sh, PTR_EXP]:bas[k_sh, PTR_EXP] + k_prim]
    al_arr = env[bas[l_sh, PTR_EXP]:bas[l_sh, PTR_EXP] + l_prim]
    ci = env[bas[i_sh, PTR_COEFF]:bas[i_sh, PTR_COEFF] + i_prim * i_ctr]
    cj = env[bas[j_sh, PTR_COEFF]:bas[j_sh, PTR_COEFF] + j_prim * j_ctr]
    ck = env[bas[k_sh, PTR_COEFF]:bas[k_sh, PTR_COEFF] + k_prim * k_ctr]
    cl = env[bas[l_sh, PTR_COEFF]:bas[l_sh, PTR_COEFF] + l_prim * l_ctr]

    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_e2 * envs.ncomp_tensor
    nc = i_ctr * j_ctr * k_ctr * l_ctr

    idx = CINTg2e_index_xyz(envs)

    expcutoff = envs.expcutoff
    dist_ij = float(np.sum((ri - rj) ** 2))
    dist_kl = float(np.sum((rk - rl) ** 2))

    gctr = np.zeros(nf * nc * n_comp)
    has_value = False
    g_alloc = envs.g_size * 3

    for lp in range(l_prim):
        envs.al = al_arr[lp]
        gctrl = np.zeros(nf * nc * n_comp) if n_comp > 1 else None
        gctrk = np.zeros(nf * i_ctr * j_ctr * k_ctr * n_comp)

        for kp in range(k_prim):
            akl = ak_arr[kp] + al_arr[lp]
            ekl = dist_kl * ak_arr[kp] * al_arr[lp] / akl
            if ekl > expcutoff:
                continue
            envs.ak = ak_arr[kp]
            envs.akl = akl
            envs.rkl = (ak_arr[kp] * rk + al_arr[lp] * rl) / akl
            envs.rklrx = envs.rkl - envs.rx_in_rklrx
            ekl_exp = math.exp(-ekl)

            gctrj = np.zeros(nf * i_ctr * j_ctr * n_comp)

            for jp in range(j_prim):
                envs.aj = aj_arr[jp]
                gctri = np.zeros(nf * i_ctr * n_comp)

                for ip in range(i_prim):
                    envs.ai = ai_arr[ip]
                    aij = ai_arr[ip] + aj_arr[jp]
                    eij = dist_ij * ai_arr[ip] * aj_arr[jp] / aij
                    if eij > expcutoff:
                        continue
                    envs.aij = aij
                    envs.rij = (ai_arr[ip] * ri + aj_arr[jp] * rj) / aij
                    envs.rijrx = envs.rij - envs.rx_in_rijrx

                    expijkl = math.exp(-eij) * ekl_exp
                    fac1i = envs.common_factor * expijkl
                    if i_ctr == 1:
                        fac1i *= ci[ip]
                    if j_ctr == 1:
                        fac1i *= cj[jp]
                    if k_ctr == 1:
                        fac1i *= ck[kp]
                    if l_ctr == 1:
                        fac1i *= cl[lp]

                    g = np.zeros(g_alloc)
                    if CINTg0_2e(g, fac1i, envs):
                        has_value = True
                        gout = np.zeros(nf * n_comp)
                        gout_2e_ar12b(gout, g, idx, envs, True)

                        # Contract i
                        if i_ctr == 1:
                            gctri[:nf * n_comp] += gout
                        else:
                            for ic in range(i_ctr):
                                c = ci[i_prim * ic + ip]
                                if c != 0:
                                    gctri[ic * nf:(ic + 1) * nf] += c * gout[:nf]

                # Contract j
                if j_ctr == 1:
                    gctrj[:nf * i_ctr * n_comp] += gctri
                else:
                    for jc in range(j_ctr):
                        c = cj[j_prim * jc + jp]
                        if c != 0:
                            sz = nf * i_ctr
                            gctrj[jc * sz:(jc + 1) * sz] += c * gctri[:sz]

            # Contract k
            if k_ctr == 1:
                gctrk[:nf * i_ctr * j_ctr * n_comp] += gctrj
            else:
                for kc in range(k_ctr):
                    c = ck[k_prim * kc + kp]
                    if c != 0:
                        sz = nf * i_ctr * j_ctr
                        gctrk[kc * sz:(kc + 1) * sz] += c * gctrj[:sz]

        # Contract l
        if l_ctr == 1:
            gctr[:nf * nc * n_comp] += gctrk
        else:
            for lc in range(l_ctr):
                c = cl[l_prim * lc + lp]
                if c != 0:
                    sz = nf * i_ctr * j_ctr * k_ctr
                    gctr[lc * sz:(lc + 1) * sz] += c * gctrk[:sz]

    return gctr, has_value


def CINT2e_spheric_drv(envs, atm, bas, env):
    """
    2e integral driver. Returns contracted spherical integrals.
    """
    x_ctr = envs.x_ctr
    i_ctr, j_ctr, k_ctr, l_ctr = [int(x) for x in x_ctr]
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_e2 * envs.ncomp_tensor

    gctr, has_value = CINT2e_loop_nopt(envs, atm, bas, env)

    if not has_value:
        di = (2 * envs.i_l + 1) * i_ctr
        dj = (2 * envs.j_l + 1) * j_ctr
        dk = (2 * envs.k_l + 1) * k_ctr
        dl = (2 * envs.l_l + 1) * l_ctr
        return np.zeros((di, dj, dk, dl))

    nfi = envs.nfi
    nfj = envs.nfj
    nfk = envs.nfk
    nfl = envs.nfl
    return c2s_sph_2e1(gctr, envs.i_l, envs.j_l, envs.k_l, envs.l_l,
                       [i_ctr, j_ctr, k_ctr, l_ctr],
                       nfi, nfj, nfk, nfl, nf)


def int2e_ar12b_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure Python int2e_ar12b_sph."""
    ng = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    envs = CINTEnvVars()
    CINTinit_int2e_EnvVars(envs, ng, shls, atm, natm, bas, nbas, env)
    envs.f_gout = gout_2e_ar12b
    result = CINT2e_spheric_drv(envs, atm, bas, env)
    # result shape: (di*i_ctr, dj*j_ctr, dk*k_ctr, dl*l_ctr)

    di_total = result.shape[0]
    dj_total = result.shape[1]
    dk_total = result.shape[2]
    dl_total = result.shape[3]

    if dims is None:
        # Block mode: write flat block into out
        block = result.ravel()
        out[:len(block)] = block
    else:
        # Full matrix mode: write at correct position using dims as strides
        naoi, naoj, naok, naol = dims[0], dims[1], dims[2], dims[3]
        for l in range(dl_total):
            for k in range(dk_total):
                for j in range(dj_total):
                    for i in range(di_total):
                        flat_idx = ((l * naok + k) * naoj + j) * naoi + i
                        out[flat_idx] = result[i, j, k, l]
    return 1


# ================================================================
# GTO driver functions
# ================================================================

def GTOint2c(intor, out, comp, hermi, shls_slice, ao_loc,
             opt, atm, natm, bas, nbas, env):
    """
    Pure Python replacement for CGTO().GTOint2c.
    Loops over shell pairs and assembles the integral matrix.
    """
    ish0, ish1, jsh0, jsh1 = shls_slice[:4]
    naoi = ao_loc[ish1] - ao_loc[ish0]
    naoj = ao_loc[jsh1] - ao_loc[jsh0]

    for ish in range(ish0, ish1):
        for jsh in range(jsh0, jsh1):
            if hermi != 0 and ish > jsh:
                continue
            shls = np.array([ish, jsh, 0, 0], dtype=np.int32)
            i0 = ao_loc[ish] - ao_loc[ish0]
            j0 = ao_loc[jsh] - ao_loc[jsh0]
            dims = [naoi, naoj]
            intor(out[j0 * naoi + i0:], dims, shls,
                  atm, atm.shape[0], bas, bas.shape[0], env, opt, None)

    # Fill lower triangle if hermitian
    if hermi != 0:
        out_mat = out.reshape(naoj, naoi)
        for i in range(naoi):
            for j in range(i + 1, naoj):
                out_mat[i, j] = out_mat[j, i]


def GTOnr2e_fill_s1(intor, eri, comp, ish_rel, jsh_rel, shls_slice,
                     ao_loc, opt, atm, natm, bas, nbas, env):
    """Fill function for 4-center integrals with s1 symmetry (no symmetry)."""
    ish0 = shls_slice[0]
    jsh0 = shls_slice[2]
    ksh0 = shls_slice[4]
    ksh1 = shls_slice[5]
    lsh0 = shls_slice[6]
    lsh1 = shls_slice[7]

    ish = ish_rel + ish0
    jsh = jsh_rel + jsh0

    naoi = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    naoj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    naok = ao_loc[shls_slice[5]] - ao_loc[shls_slice[4]]
    naol = ao_loc[shls_slice[7]] - ao_loc[shls_slice[6]]

    i0 = ao_loc[ish] - ao_loc[ish0]
    j0 = ao_loc[jsh] - ao_loc[jsh0]

    di = ao_loc[ish + 1] - ao_loc[ish]
    dj = ao_loc[jsh + 1] - ao_loc[jsh]
    eri_4d = eri.reshape(naoi, naoj, naok, naol)

    for ksh in range(ksh0, ksh1):
        for lsh in range(lsh0, lsh1):
            k0 = ao_loc[ksh] - ao_loc[ksh0]
            l0 = ao_loc[lsh] - ao_loc[lsh0]
            dk = ao_loc[ksh + 1] - ao_loc[ksh]
            dl = ao_loc[lsh + 1] - ao_loc[lsh]

            shls = np.array([ish, jsh, ksh, lsh], dtype=np.int32)
            buf = np.zeros(di * dj * dk * dl)
            intor(buf, None, shls, atm, atm.shape[0], bas, bas.shape[0], env, opt, None)

            eri_4d[i0:i0+di, j0:j0+dj, k0:k0+dk, l0:l0+dl] = \
                buf.reshape(di, dj, dk, dl)


def GTOnr2e_fill_drv(intor, fill, eri, comp, shls_slice, ao_loc,
                      opt, atm, natm, bas, nbas, env):
    """
    Pure Python replacement for CGTO().GTOnr2e_fill_drv.
    """
    ish0, ish1, jsh0, jsh1 = shls_slice[:4]
    nish = ish1 - ish0
    njsh = jsh1 - jsh0

    for ij in range(nish * njsh):
        i = ij // njsh
        j = ij % njsh
        fill(intor, eri, comp, i, j, shls_slice,
             ao_loc, opt, atm, natm, bas, nbas, env)


# ================================================================
# 3-center 2-electron integrals
# ================================================================

def CINT3c2e_loop_nopt(envs, atm, bas, env):
    """
    3c2e integral primitive loop (no optimizer).
    Like CINT2e_loop_nopt but with only 3 shells (i,j,k) and no l-shell.
    Returns (gctr, has_value).
    """
    shls = envs.shls
    i_sh, j_sh, k_sh = int(shls[0]), int(shls[1]), int(shls[2])
    i_ctr = int(envs.x_ctr[0])
    j_ctr = int(envs.x_ctr[1])
    k_ctr = int(envs.x_ctr[2])
    i_prim = int(bas[i_sh, NPRIM_OF])
    j_prim = int(bas[j_sh, NPRIM_OF])
    k_prim = int(bas[k_sh, NPRIM_OF])

    ri, rj = envs.ri, envs.rj
    ai_arr = env[bas[i_sh, PTR_EXP]:bas[i_sh, PTR_EXP] + i_prim]
    aj_arr = env[bas[j_sh, PTR_EXP]:bas[j_sh, PTR_EXP] + j_prim]
    ak_arr = env[bas[k_sh, PTR_EXP]:bas[k_sh, PTR_EXP] + k_prim]
    ci = env[bas[i_sh, PTR_COEFF]:bas[i_sh, PTR_COEFF] + i_prim * i_ctr]
    cj = env[bas[j_sh, PTR_COEFF]:bas[j_sh, PTR_COEFF] + j_prim * j_ctr]
    ck = env[bas[k_sh, PTR_COEFF]:bas[k_sh, PTR_COEFF] + k_prim * k_ctr]

    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor  # note: no ncomp_e2 for 3c
    nc = i_ctr * j_ctr * k_ctr

    idx = CINTg2e_index_xyz(envs)

    expcutoff = envs.expcutoff
    dist_ij = float(np.sum((ri - rj) ** 2))

    gctr = np.zeros(nf * nc * n_comp)
    has_value = False
    g_alloc = envs.g_size * 3

    for kp in range(k_prim):
        envs.ak = ak_arr[kp]
        envs.akl = ak_arr[kp]  # al=0, so akl = ak

        gctrj = np.zeros(nf * i_ctr * j_ctr * n_comp)

        for jp in range(j_prim):
            envs.aj = aj_arr[jp]
            gctri = np.zeros(nf * i_ctr * n_comp)

            for ip in range(i_prim):
                envs.ai = ai_arr[ip]
                aij = ai_arr[ip] + aj_arr[jp]
                eij = dist_ij * ai_arr[ip] * aj_arr[jp] / aij
                if eij > expcutoff:
                    continue
                envs.aij = aij
                envs.rij = (ai_arr[ip] * ri + aj_arr[jp] * rj) / aij
                envs.rijrx = envs.rij - envs.rx_in_rijrx

                expij = math.exp(-eij)
                fac1i = envs.common_factor * expij
                if i_ctr == 1:
                    fac1i *= ci[ip]
                if j_ctr == 1:
                    fac1i *= cj[jp]
                if k_ctr == 1:
                    fac1i *= ck[kp]

                g = np.zeros(g_alloc)
                if CINTg0_2e(g, fac1i, envs):
                    has_value = True
                    gout = np.zeros(nf * n_comp)
                    gout_2e_ar12b(gout, g, idx, envs, True)

                    # Contract i
                    if i_ctr == 1:
                        gctri[:nf * n_comp] += gout
                    else:
                        for ic in range(i_ctr):
                            c = ci[i_prim * ic + ip]
                            if c != 0:
                                gctri[ic * nf:(ic + 1) * nf] += c * gout[:nf]

            # Contract j
            if j_ctr == 1:
                gctrj[:nf * i_ctr * n_comp] += gctri
            else:
                for jc in range(j_ctr):
                    c = cj[j_prim * jc + jp]
                    if c != 0:
                        off_j = jc * nf * i_ctr
                        gctrj[off_j:off_j + nf * i_ctr] += c * gctri[:nf * i_ctr]

        # Contract k
        if k_ctr == 1:
            gctr[:nf * nc * n_comp] += gctrj
        else:
            for kc in range(k_ctr):
                c = ck[k_prim * kc + kp]
                if c != 0:
                    off_k = kc * nf * i_ctr * j_ctr
                    gctrj_size = nf * i_ctr * j_ctr
                    gctr[off_k:off_k + gctrj_size] += c * gctrj[:gctrj_size]

    return gctr, has_value


def c2s_sph_3c2e1(gctr, i_l, j_l, k_l, x_ctr, nfi, nfj, nfk, nf):
    """Cart-to-spherical transformation for 3c2e integrals.
    Returns array of shape (di*i_ctr, dj*j_ctr, dk*k_ctr)."""
    i_ctr, j_ctr, k_ctr = x_ctr[0], x_ctr[1], x_ctr[2]
    di = 2 * i_l + 1
    dj = 2 * j_l + 1
    dk = 2 * k_l + 1

    c2s_i = cart2sph_matrix(i_l)
    c2s_j = cart2sph_matrix(j_l)
    c2s_k = cart2sph_matrix(k_l)

    nc = i_ctr * j_ctr * k_ctr
    # gctr layout: (nc, nf) where nf = nfi * nfj * nfk
    gctr_3d = gctr.reshape(nc, nf)

    out = np.zeros((di * i_ctr, dj * j_ctr, dk * k_ctr))

    n = 0
    for kc in range(k_ctr):
        for jc in range(j_ctr):
            for ic in range(i_ctr):
                # gctr layout is (nfj, nfk, nfi) in Fortran order from
                # the contraction loop order j->l(k)->k(unused)->i
                block = gctr_3d[n].reshape(nfj, nfk, nfi)
                # Transform j: (nfj, nfk, nfi) -> (dj, nfk, nfi)
                tmp = np.tensordot(c2s_j, block, axes=([1], [0]))
                # Transform k: (dj, nfk, nfi) -> (dj, dk, nfi)
                tmp = np.tensordot(tmp, c2s_k.T, axes=([1], [0]))
                # Transform i: (dj, dk, nfi) -> (dj, dk, di)
                tmp = np.tensordot(tmp, c2s_i.T, axes=([2], [0]))
                # Reorder to (di, dj, dk)
                out[ic*di:(ic+1)*di, jc*dj:(jc+1)*dj, kc*dk:(kc+1)*dk] = tmp.transpose(2, 0, 1)
                n += 1

    return out


def CINT3c2e_spheric_drv(envs, atm, bas, env):
    """3c2e integral driver. Returns contracted spherical integrals."""
    x_ctr = envs.x_ctr
    i_ctr, j_ctr, k_ctr = int(x_ctr[0]), int(x_ctr[1]), int(x_ctr[2])
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor

    gctr, has_value = CINT3c2e_loop_nopt(envs, atm, bas, env)

    if not has_value:
        di = (2 * envs.i_l + 1) * i_ctr
        dj = (2 * envs.j_l + 1) * j_ctr
        dk = (2 * envs.k_l + 1) * k_ctr
        return np.zeros((di, dj, dk))

    nfi = envs.nfi
    nfj = envs.nfj
    nfk = envs.nfk
    return c2s_sph_3c2e1(gctr, envs.i_l, envs.j_l, envs.k_l,
                          [i_ctr, j_ctr, k_ctr], nfi, nfj, nfk, nf)


def int3c2e_ar12_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure Python int3c2e_ar12_sph."""
    ng = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    envs = CINTEnvVars()
    CINTinit_int3c2e_EnvVars(envs, ng, shls, atm, natm, bas, nbas, env)
    envs.f_gout = gout_2e_ar12b
    result = CINT3c2e_spheric_drv(envs, atm, bas, env)
    # result shape: (di*i_ctr, dj*j_ctr, dk*k_ctr)

    if dims is None:
        # Block mode: write flat block into out
        block = result.ravel()
        out[:len(block)] = block
    else:
        # Full matrix mode: write at correct position using dims as strides
        naoi, naoj, naok = dims[0], dims[1], dims[2]
        di_total, dj_total, dk_total = result.shape
        for k in range(dk_total):
            for j in range(dj_total):
                for i in range(di_total):
                    flat_idx = (k * naoj + j) * naoi + i
                    out[flat_idx] = result[i, j, k]
    return 1


def GTOnr3c_fill_s1(intor, out, buf, comp, jobid, shls_slice, ao_loc,
                     opt, atm, natm, bas, nbas, env):
    """Fill function for 3-center integrals with s1 symmetry."""
    BLKSIZE = 8
    ish0 = shls_slice[0]
    ish1 = shls_slice[1]
    jsh0 = shls_slice[2]
    jsh1 = shls_slice[3]
    ksh0 = shls_slice[4]
    ksh1 = shls_slice[5]
    nksh = ksh1 - ksh0

    ksh = jobid % nksh + ksh0
    jstart = (jobid // nksh) * BLKSIZE + jsh0
    jend = min(jstart + BLKSIZE, jsh1)
    if jstart >= jend:
        return

    naoi = ao_loc[ish1] - ao_loc[ish0]
    naoj = ao_loc[jsh1] - ao_loc[jsh0]
    naok = ao_loc[ksh1] - ao_loc[ksh0]
    dims = [naoi, naoj, naok]

    k0 = ao_loc[ksh] - ao_loc[ksh0]
    out_offset = naoi * naoj * k0

    for jsh in range(jstart, jend):
        for ish in range(ish0, ish1):
            shls = np.array([ish, jsh, ksh, 0], dtype=np.int32)
            i0 = ao_loc[ish] - ao_loc[ish0]
            j0 = ao_loc[jsh] - ao_loc[jsh0]
            intor(out[out_offset + j0 * naoi + i0:], dims, shls,
                  atm, natm, bas, nbas, env, opt, None)


def GTOnr3c_drv(intor, fill, eri, comp, shls_slice, ao_loc,
                 opt, atm, natm, bas, nbas, env):
    """Pure Python replacement for CGTO().GTOnr3c_drv."""
    BLKSIZE = 8
    ish0, ish1 = shls_slice[0], shls_slice[1]
    jsh0, jsh1 = shls_slice[2], shls_slice[3]
    ksh0, ksh1 = shls_slice[4], shls_slice[5]
    nish = ish1 - ish0
    njsh = jsh1 - jsh0
    nksh = ksh1 - ksh0

    njobs = ((max(nish, njsh) + BLKSIZE - 1) // BLKSIZE) * nksh

    for jobid in range(njobs):
        fill(intor, eri, None, comp, jobid, shls_slice, ao_loc,
             opt, atm, natm, bas, nbas, env)


# ================================================================
# Fourier Transform of GTO basis functions
# ================================================================

def _ft_1d_poly(k, n, a2):
    """Compute the 1D polynomial P_n(-ik*a2) for FT of x^n * exp(-alpha*x^2).

    The recurrence is:
        P_0 = 1
        P_1 = -ik*a2
        P_{n+1} = n*a2 * P_{n-1} + (-ik*a2) * P_n

    Parameters
    ----------
    k : ndarray, shape (NGv,)
        G-vector component.
    n : int
        Power of the coordinate.
    a2 : float
        0.5 / alpha.

    Returns
    -------
    ndarray, shape (NGv,), complex
    """
    if n == 0:
        return np.ones(len(k), dtype=np.complex128)
    ikha = -1j * k * a2
    if n == 1:
        return ikha
    # General recurrence
    p_prev = np.ones(len(k), dtype=np.complex128)
    p_curr = ikha.copy()
    for m in range(1, n):
        p_next = m * a2 * p_prev + ikha * p_curr
        p_prev = p_curr
        p_curr = p_next
    return p_curr


def gto_ft_evaluator_py(wrapper, gvgrid):
    """Pure Python FT evaluator for GTO basis functions.

    Computes FT(phi_i)(G) = integral phi_i(r) * exp(-iG.r) dr
    for each AO i and G-vector.

    Uses the analytical formula for the FT of Cartesian GTOs:
        FT(x^a y^b z^c exp(-alpha*r^2))(G) = prod_d f(G_d, l_d, alpha)
    where f(k, n, alpha) = sqrt(pi/alpha) * exp(-k^2/(4*alpha)) * P_n(-ik/(2*alpha))

    Parameters
    ----------
    wrapper : LibcintWrapper
        Basis set wrapper.
    gvgrid : torch.Tensor, shape (NGv, 3)
        G-vectors at which to evaluate the FT.

    Returns
    -------
    ndarray, shape (nao, NGv), complex128
    """
    from deepchem.utils.analytical_integrators.optimizer import (
        CINTcommon_fac_sp, ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF,
        PTR_EXP, PTR_COEFF, PTR_COORD,
    )

    atm, bas, env = wrapper.atm_bas_env
    ao_loc = wrapper.full_shell_to_aoloc
    nao = wrapper.nao()
    Gv = np.asarray(gvgrid.detach().cpu().numpy(), dtype=np.float64)
    NGv = Gv.shape[0]
    Gx, Gy, Gz = Gv[:, 0], Gv[:, 1], Gv[:, 2]
    G2 = Gx**2 + Gy**2 + Gz**2

    out = np.zeros((nao, NGv), dtype=np.complex128)

    ish0, ish1 = wrapper.shell_idxs
    for ish in range(ish0, ish1):
        l = int(bas[ish, ANG_OF])
        nprim = int(bas[ish, NPRIM_OF])
        nctr = int(bas[ish, NCTR_OF])
        nf_cart = (l + 1) * (l + 2) // 2
        di = 2 * l + 1  # spherical components

        atom_idx = int(bas[ish, ATOM_OF])
        R = env[atm[atom_idx, PTR_COORD]: atm[atom_idx, PTR_COORD] + 3]
        alphas = env[bas[ish, PTR_EXP]: bas[ish, PTR_EXP] + nprim]
        coeffs = env[bas[ish, PTR_COEFF]: bas[ish, PTR_COEFF] + nprim * nctr]

        # Phase factor exp(-iG.R) for this shell
        phase = np.exp(-1j * (Gx * R[0] + Gy * R[1] + Gz * R[2]))

        # Cartesian component indices
        i_nx, i_ny, i_nz = CINTcart_comp(l)

        # Normalization factor matching C code:
        # fac1 = sqrt(pi) * pi * CINTcommon_fac_sp(l) * CINTcommon_fac_sp(0)
        # ghost coeff = sqrt(4*pi)
        # dij = 1 / (alpha^(3/2))
        # Total per-primitive = ci * ghost_c * fac1 * dij
        fac_norm = (math.sqrt(math.pi) * math.pi
                    * CINTcommon_fac_sp(l) * CINTcommon_fac_sp(0)
                    * math.sqrt(4.0 * math.pi))

        # For each contraction
        for ic in range(nctr):
            # Accumulate Cartesian FT over primitives
            cart_ft = np.zeros((nf_cart, NGv), dtype=np.complex128)

            for ip in range(nprim):
                alpha = alphas[ip]
                c = coeffs[ic * nprim + ip]
                if abs(c) < 1e-30:
                    continue

                a2 = 0.5 / alpha
                # Radial factor: fac_norm * c / alpha^(3/2) * exp(-G^2/(4*alpha))
                radial = fac_norm * c / (alpha * math.sqrt(alpha))
                exp_factor = np.exp(-G2 * a2 * 0.5)  # exp(-G^2/(4*alpha))
                base = radial * exp_factor * phase  # shape (NGv,)

                # Build 1D polynomials for each needed power
                max_n = l
                px = {}
                py = {}
                pz = {}
                for n in range(max_n + 1):
                    px[n] = _ft_1d_poly(Gx, n, a2)
                    py[n] = _ft_1d_poly(Gy, n, a2)
                    pz[n] = _ft_1d_poly(Gz, n, a2)

                # Compute Cartesian FT components
                for f in range(nf_cart):
                    a, b, cc = i_nx[f], i_ny[f], i_nz[f]
                    cart_ft[f] += base * px[a] * py[b] * pz[cc]

            # Transform from Cartesian to spherical
            c2s = cart2sph_matrix(l)  # shape (di, nf_cart)
            sph_ft = c2s @ cart_ft  # shape (di, NGv)

            # Place into output
            ao_start = ao_loc[ish] - ao_loc[ish0] + ic * di
            out[ao_start: ao_start + di, :] = sph_ft

    return out


# ================================================================
# GTO grid evaluator (replaces CGTO().GTOval_*_sph/cart)
# ================================================================

def gto_evaluator_py_grid(wrapper, shortname, rgrid, spherical):
    """Pure Python GTO grid evaluator.

    Replaces CGTO().GTOval_sph, GTOval_ip_sph, GTOval_lapl_sph,
    GTOval_rr_sph, and their _cart variants.

    Parameters
    ----------
    wrapper : LibcintWrapper
        Basis set wrapper.
    shortname : str
        Operation type: "" (value), "ip" (gradient), "lapl" (Laplacian),
        "rr" (value * r^2).
    rgrid : torch.Tensor, shape (ngrid, 3)
        Real-space grid points.
    spherical : bool
        If True, use spherical harmonics; if False, Cartesian.

    Returns
    -------
    ndarray
        Shape (nao, ngrid) for "" / "lapl" / "rr",
        or (3, nao, ngrid) for "ip".
    """
    import re

    atm, bas, env = wrapper.atm_bas_env
    ao_loc = np.asarray(wrapper.full_shell_to_aoloc, dtype=np.int32)
    nao = wrapper.nao()
    coords = np.asarray(rgrid.detach().cpu().numpy(), dtype=np.float64)
    ngrid = coords.shape[0]

    n_ip = len(re.findall(r"^(?:ip)*(?:ip)?", shortname)[0]) // 2
    comp_shape = (3,) * n_ip
    ncomp = 3 ** n_ip if n_ip > 0 else 1

    if ncomp > 1:
        out = np.zeros(comp_shape + (nao, ngrid), dtype=np.float64)
    else:
        out = np.zeros((nao, ngrid), dtype=np.float64)

    ish0, ish1 = wrapper.shell_idxs

    for ish in range(ish0, ish1):
        l = int(bas[ish, ANG_OF])
        nprim = int(bas[ish, NPRIM_OF])
        nctr = int(bas[ish, NCTR_OF])
        atom_idx = int(bas[ish, ATOM_OF])
        nf_cart = (l + 1) * (l + 2) // 2
        di = 2 * l + 1 if spherical else nf_cart

        R = env[atm[atom_idx, PTR_COORD]: atm[atom_idx, PTR_COORD] + 3]
        alphas_arr = env[bas[ish, PTR_EXP]: bas[ish, PTR_EXP] + nprim]
        coeffs_flat = env[bas[ish, PTR_COEFF]: bas[ish, PTR_COEFF] + nprim * nctr]

        fac = CINTcommon_fac_sp(l)

        # Relative coordinates (r - R)
        rx = coords[:, 0] - R[0]
        ry = coords[:, 1] - R[1]
        rz = coords[:, 2] - R[2]
        r2 = rx * rx + ry * ry + rz * rz

        # Primitive exponentials: (nprim, ngrid)
        eprim = np.exp(-alphas_arr[:, None] * r2[None, :]) * fac

        # Max coordinate power needed
        max_pow = l
        if 'ip' in shortname:
            max_pow = max(max_pow, l + 1)
        if 'lapl' in shortname or 'rr' in shortname:
            max_pow = max(max_pow, l + 2)

        # Pre-compute coordinate powers
        xpows = np.ones((max_pow + 1, ngrid))
        ypows = np.ones((max_pow + 1, ngrid))
        zpows = np.ones((max_pow + 1, ngrid))
        for p in range(1, max_pow + 1):
            xpows[p] = xpows[p - 1] * rx
            ypows[p] = ypows[p - 1] * ry
            zpows[p] = zpows[p - 1] * rz

        # Cartesian component ordering (lx descending, matching C code)
        cart_indices = []
        for lx in range(l, -1, -1):
            for ly in range(l - lx, -1, -1):
                lz = l - lx - ly
                cart_indices.append((lx, ly, lz))

        for ic in range(nctr):
            coeff_ic = coeffs_flat[ic * nprim: (ic + 1) * nprim]

            if shortname == "":
                ectr = coeff_ic @ eprim
                cart_vals = np.empty((nf_cart, ngrid))
                for f, (lx, ly, lz) in enumerate(cart_indices):
                    cart_vals[f] = ectr * xpows[lx] * ypows[ly] * zpows[lz]

            elif shortname == "ip":
                ectr = coeff_ic @ eprim
                ectr_2a = ((-2 * alphas_arr) * coeff_ic) @ eprim
                cart_vals = np.zeros((3, nf_cart, ngrid))
                for f, (lx, ly, lz) in enumerate(cart_indices):
                    yz = ypows[ly] * zpows[lz]
                    xz = xpows[lx] * zpows[lz]
                    xy = xpows[lx] * ypows[ly]
                    dx_val = ectr_2a * xpows[lx + 1]
                    if lx > 0:
                        dx_val = dx_val + ectr * lx * xpows[lx - 1]
                    cart_vals[0, f] = dx_val * yz
                    dy_val = ectr_2a * ypows[ly + 1]
                    if ly > 0:
                        dy_val = dy_val + ectr * ly * ypows[ly - 1]
                    cart_vals[1, f] = dy_val * xz
                    dz_val = ectr_2a * zpows[lz + 1]
                    if lz > 0:
                        dz_val = dz_val + ectr * lz * zpows[lz - 1]
                    cart_vals[2, f] = dz_val * xy

            elif shortname == "lapl":
                ectr = coeff_ic @ eprim
                ectr_2a = ((-2 * alphas_arr) * coeff_ic) @ eprim
                ectr_4a2 = ((4 * alphas_arr**2) * coeff_ic) @ eprim
                cart_vals = np.zeros((nf_cart, ngrid))
                for f, (lx, ly, lz) in enumerate(cart_indices):
                    yz = ypows[ly] * zpows[lz]
                    xz = xpows[lx] * zpows[lz]
                    xy = xpows[lx] * ypows[ly]
                    d2x = ((2 * lx + 1) * xpows[lx] * ectr_2a
                           + xpows[lx + 2] * ectr_4a2)
                    if lx >= 2:
                        d2x = d2x + lx * (lx - 1) * xpows[lx - 2] * ectr
                    d2y = ((2 * ly + 1) * ypows[ly] * ectr_2a
                           + ypows[ly + 2] * ectr_4a2)
                    if ly >= 2:
                        d2y = d2y + ly * (ly - 1) * ypows[ly - 2] * ectr
                    d2z = ((2 * lz + 1) * zpows[lz] * ectr_2a
                           + zpows[lz + 2] * ectr_4a2)
                    if lz >= 2:
                        d2z = d2z + lz * (lz - 1) * zpows[lz - 2] * ectr
                    cart_vals[f] = d2x * yz + xz * d2y + xy * d2z

            elif shortname == "rr":
                ectr = coeff_ic @ eprim
                cart_vals = np.empty((nf_cart, ngrid))
                for f, (lx, ly, lz) in enumerate(cart_indices):
                    cart_vals[f] = (ectr * r2
                                   * xpows[lx] * ypows[ly] * zpows[lz])

            else:
                raise NotImplementedError(
                    "GTO grid evaluation for shortname='%s' not implemented"
                    % shortname)

            # Cartesian to spherical transform
            if spherical and l >= 2:
                c2s = cart2sph_matrix(l)
                if ncomp > 1:
                    sph = np.empty(comp_shape + (di, ngrid))
                    for idx in np.ndindex(comp_shape):
                        sph[idx] = c2s @ cart_vals[idx]
                else:
                    sph = c2s @ cart_vals
            else:
                sph = cart_vals

            ao_start = ao_loc[ish] - ao_loc[ish0] + ic * di
            if ncomp > 1:
                out[..., ao_start:ao_start + di, :] = sph
            else:
                out[ao_start:ao_start + di, :] = sph

    return out


# ================================================================
# Integral function registry (maps opname -> Python function)
# ================================================================

INTEGRAL_REGISTRY = {
    'int1e_ovlp_sph': int1e_ovlp_sph,
    'int1e_kin_sph': int1e_kin_sph,
    'int1e_nuc_sph': int1e_nuc_sph,
    'int2e_ar12b_sph': int2e_ar12b_sph,
    'int3c2e_ar12_sph': int3c2e_ar12_sph,
}
