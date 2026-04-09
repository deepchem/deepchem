"""
Pure PyTorch implementations of libcint integral functions.

Replaces the C libcint functions:
  - compute_overlap_1e_sph  (overlap)
  - compute_kinetic_1e_sph   (kinetic energy)
  - compute_nuclear_1e_sph   (nuclear attraction)
  - compute_eri_2e_sph (electron repulsion)

And the GTO driver functions:
  - assemble_2center_integrals        (2-center integral matrix assembly)
  - fill_4center_s1 / fill_4center_driver (4-center integral assembly)
"""
import math
import torch
from deepchem.utils.analytical_integrators_torch.optimizer import (
    CINTEnvVars, cartesian_components, sph_harmonic_norm,
    init_envvars_1e, compute_g_index_1e,
    init_envvars_2e, compute_g_index_2e,
    init_envvars_3c2e,
    ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF,
    PTR_COORD,
)
from deepchem.utils.analytical_integrators_torch.spherical import rys_roots

SQRTPI = math.sqrt(math.pi)

# Cartesian-to-spherical transformation matrices

def _cart_to_sph_matrix(l):
    """Return the (2l+1, nf_cart) transformation matrix from Cartesian
    to real solid harmonics for angular momentum l."""
    if l == 0:
        return torch.tensor([[1.0]], dtype=torch.float64)
    elif l == 1:
        return torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64)
    elif l == 2:
        s3 = math.sqrt(3.0)
        return torch.tensor([
            [0.0,      s3,       0.0,     0.0,     0.0,     0.0     ],
            [0.0,      0.0,      0.0,     0.0,     s3,      0.0     ],
            [-0.5,     0.0,      0.0,    -0.5,     0.0,     1.0     ],
            [0.0,      0.0,      s3,      0.0,     0.0,     0.0     ],
            [s3/2.0,   0.0,      0.0,    -s3/2.0,  0.0,     0.0     ],
        ], dtype=torch.float64)
    elif l == 3:
        s = math.sqrt
        return torch.tensor([
            [0,         s(10)/4, 0,       0,       0,       0,       -s(10)/4*3,0,       0,       0        ],
            [0,         0,       0,       0,       s(15),   0,       0,        0,        0,       0        ],
            [0,         -s(6)/4, 0,       0,       0,       0,       -s(6)/4,  0,        s(6),    0        ],
            [0,         0,       -3./2.,  0,       0,       0,       0,        -3./2.,   0,       1.       ],
            [-s(6)/4,   0,       0,       -s(6)/4, 0,       s(6)/2,  0,        0,        0,       0        ],
            [0,         0,       s(15)/2, 0,       0,       0,       0,        -s(15)/2, 0,       0        ],
            [s(10)/4*3, 0,       0,       -s(10)/4,0,       0,       0,        0,        0,       0        ],
        ], dtype=torch.float64)
    else:
        raise NotImplementedError(f"cart2sph not implemented for l={l}")


_CART2SPH_CACHE = {}

def cart_to_sph_matrix(l):
    """Get cached cart2sph transformation matrix for angular momentum l."""
    if l not in _CART2SPH_CACHE:
        _CART2SPH_CACHE[l] = _cart_to_sph_matrix(l)
    return _CART2SPH_CACHE[l]


def cart_to_sph_1e(gctr, i_l, j_l, i_ctr, j_ctr, nfi, nfj, nf):
    """Cartesian-to-spherical transformation for 1e integrals."""
    di = 2 * i_l + 1
    dj = 2 * j_l + 1

    c2s_i = cart_to_sph_matrix(i_l)
    c2s_j = cart_to_sph_matrix(j_l)

    out = torch.zeros((di * i_ctr, dj * j_ctr), dtype=torch.float64)

    for jc in range(j_ctr):
        for ic in range(i_ctr):
            offset = (jc * i_ctr + ic) * nf
            cart_block = gctr[offset:offset + nf].reshape(nfj, nfi)
            sph_block = c2s_i @ cart_block.T @ c2s_j.T
            i0 = ic * di
            j0 = jc * dj
            out[i0:i0 + di, j0:j0 + dj] = sph_block

    return out


def cart_to_sph_2e(gctr, i_l, j_l, k_l, l_l, x_ctr, nfi, nfj, nfk, nfl, nf):
    """Cartesian-to-spherical transformation for 2e integrals."""
    i_ctr, j_ctr, k_ctr, l_ctr = x_ctr
    di = 2 * i_l + 1
    dj = 2 * j_l + 1
    dk = 2 * k_l + 1
    dl = 2 * l_l + 1

    c2s_i = cart_to_sph_matrix(i_l)
    c2s_j = cart_to_sph_matrix(j_l)
    c2s_k = cart_to_sph_matrix(k_l)
    c2s_l = cart_to_sph_matrix(l_l)

    out = torch.zeros((di * i_ctr, dj * j_ctr, dk * k_ctr, dl * l_ctr), dtype=torch.float64)

    for lc in range(l_ctr):
        for kc in range(k_ctr):
            for jc in range(j_ctr):
                for ic in range(i_ctr):
                    idx = ((lc * k_ctr + kc) * j_ctr + jc) * i_ctr + ic
                    offset = idx * nf
                    cart_4d = gctr[offset:offset + nf].reshape(nfj, nfl, nfk, nfi)
                    tmp = torch.tensordot(c2s_i, cart_4d, dims=([1], [3]))
                    tmp = torch.tensordot(c2s_j, tmp, dims=([1], [1]))
                    tmp = torch.tensordot(c2s_k, tmp, dims=([1], [3]))
                    tmp = torch.tensordot(c2s_l, tmp, dims=([1], [3]))
                    i0, j0, k0, l0 = ic * di, jc * dj, kc * dk, lc * dl
                    out[i0:i0+di, j0:j0+dj, k0:k0+dk, l0:l0+dl] = tmp.permute(3, 2, 1, 0)

    return out


# G-value generation (recurrence relations)

def _g_vertical_horizontal_recurrence(g, envs, gz0_fac, rir0, cfac, aij):
    """Unified vertical + horizontal recurrence for 1e g-values."""
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


def compute_g_overlap(g, ai, aj, fac, envs):
    """Generate g-values for overlap integrals."""
    aij = ai + aj
    rir0 = envs.ri - (ai * envs.ri + aj * envs.rj) / aij
    _g_vertical_horizontal_recurrence(g, envs, SQRTPI * math.pi * fac, rir0, 0.5 / aij, aij)


def compute_g_nuclear(g, aij, rij, cr, t2, fac, envs):
    """Generate g-values for nuclear attraction integrals."""
    rir0 = envs.ri - (rij + t2 * (cr - rij))
    _g_vertical_horizontal_recurrence(g, envs, 2.0 * math.pi * fac, rir0, 0.5 * (1.0 - t2) / aij, aij)


# Nabla (derivative) operators for kinetic energy

def apply_nabla_j_1e(f, g, li, lj, lk, envs):
    """Compute nabla_j g-values for kinetic energy integrals."""
    dj = envs.g_stride_j
    aj2 = -2.0 * envs.aj
    g_size = envs.g_size
    gx, gy, gz = g[:g_size], g[g_size:2*g_size], g[2*g_size:3*g_size]
    fx, fy, fz = f[:g_size], f[g_size:2*g_size], f[2*g_size:3*g_size]

    for k in range(lk + 1):
        dk_off = 0
        ptr = dk_off
        for i in range(ptr, ptr + li + 1):
            fx[i] = aj2 * gx[i + dj]
            fy[i] = aj2 * gy[i + dj]
            fz[i] = aj2 * gz[i + dj]
        for j in range(1, lj + 1):
            ptr = dj * j + dk_off
            for i in range(ptr, ptr + li + 1):
                fx[i] = j * gx[i - dj] + aj2 * gx[i + dj]
                fy[i] = j * gy[i - dj] + aj2 * gy[i + dj]
                fz[i] = j * gz[i - dj] + aj2 * gz[i + dj]


# Gout extraction functions

def extract_gout_overlap(gout, g, idx, envs):
    """Extract overlap gout using vectorized indexing."""
    ix = idx[0::3].long()
    iy = idx[1::3].long()
    iz = idx[2::3].long()
    gout[:envs.nf] += g[ix] * g[iy] * g[iz]


def extract_gout_nuclear(gout, g, idx, envs):
    """Extract nuclear attraction gout (same as overlap)."""
    extract_gout_overlap(gout, g, idx, envs)


def extract_gout_kinetic(gout, g, idx, envs):
    """Extract kinetic energy gout: <i| -1/2 nabla^2 |j>."""
    g_size = envs.g_size
    g_len = g_size * 3
    g0 = g[:g_len]
    g1 = g[g_len:2 * g_len]
    g2 = g[2 * g_len:3 * g_len]
    g3 = g[3 * g_len:4 * g_len]

    apply_nabla_j_1e(g1, g0, envs.i_l, envs.j_l, 0, envs)
    apply_nabla_j_1e(g2, g0, envs.i_l, envs.j_l + 1, 0, envs)
    apply_nabla_j_1e(g3, g2, envs.i_l, envs.j_l, 0, envs)

    ix = idx[0::3].long()
    iy = idx[1::3].long()
    iz = idx[2::3].long()
    gout[:envs.nf] += -(g3[ix] * g0[iy] * g0[iz] +
                         g0[ix] * g3[iy] * g0[iz] +
                         g0[ix] * g0[iy] * g3[iz])


def extract_gout_2e(gout, g, idx, envs, gout_empty):
    """Extract 2e gout: sum over Rys roots of g[ix]*g[iy]*g[iz]."""
    nroots = envs.nrys_roots
    ix = idx[0::3].long()
    iy = idx[1::3].long()
    iz = idx[2::3].long()
    s = torch.zeros(envs.nf, dtype=torch.float64)
    for i in range(nroots):
        s += g[ix + i] * g[iy + i] * g[iz + i]
    if gout_empty:
        gout[:envs.nf] = s
    else:
        gout[:envs.nf] += s


# 1e integral loops

def primitive_loop_1e(envs, atm, bas, env):
    """1e overlap/kinetic primitive loop."""
    shls = envs.shls
    i_sh, j_sh = int(shls[0].item()), int(shls[1].item())
    i_l, j_l = envs.i_l, envs.j_l
    i_ctr, j_ctr = int(envs.x_ctr[0].item()), int(envs.x_ctr[1].item())
    i_prim = int(bas[i_sh, NPRIM_OF].item())
    j_prim = int(bas[j_sh, NPRIM_OF].item())
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor
    ri, rj = envs.ri, envs.rj
    ai_arr = env[int(bas[i_sh, PTR_EXP].item()):int(bas[i_sh, PTR_EXP].item()) + i_prim]
    aj_arr = env[int(bas[j_sh, PTR_EXP].item()):int(bas[j_sh, PTR_EXP].item()) + j_prim]
    ci = env[int(bas[i_sh, PTR_COEFF].item()):int(bas[i_sh, PTR_COEFF].item()) + i_prim * i_ctr]
    cj = env[int(bas[j_sh, PTR_COEFF].item()):int(bas[j_sh, PTR_COEFF].item()) + j_prim * j_ctr]

    idx = compute_g_index_1e(envs)
    rrij = ((ri - rj) ** 2).sum()
    fac = envs.common_factor * sph_harmonic_norm(i_l) * sph_harmonic_norm(j_l)
    expcutoff = envs.expcutoff

    nc = nf * i_ctr * j_ctr
    gctr = torch.zeros(nc * n_comp, dtype=torch.float64)
    has_value = False

    gbits = envs.gbits
    g_sections = max((1 << gbits) + 1, 4)
    g_alloc = envs.g_size * 3 * g_sections

    for jp in range(j_prim):
        envs.aj = aj_arr[jp].item()
        gctri = torch.zeros(nf * i_ctr * n_comp, dtype=torch.float64)

        for ip in range(i_prim):
            envs.ai = ai_arr[ip].item()
            aij = ai_arr[ip] + aj_arr[jp]
            eij = (ai_arr[ip] * aj_arr[jp] / aij) * rrij
            if eij.item() > expcutoff:
                continue
            has_value = True

            dij = torch.exp(-eij) / (aij * torch.sqrt(aij)) * fac

            g = torch.zeros(g_alloc, dtype=torch.float64)
            compute_g_overlap(g, ai_arr[ip], aj_arr[jp], dij, envs)

            gout = torch.zeros(nf * n_comp, dtype=torch.float64)
            envs.f_gout(gout, g, idx, envs)

            for n in range(n_comp):
                block = gout[n * nf:(n + 1) * nf]
                for ic in range(i_ctr):
                    c = ci[i_prim * ic + ip]
                    if c.item() != 0:
                        offset = n * nf * i_ctr + ic * nf
                        gctri[offset:offset + nf] = gctri[offset:offset + nf] + c * block

        for n in range(n_comp):
            for jc in range(j_ctr):
                c = cj[j_prim * jc + jp]
                if c.item() != 0:
                    for ic in range(i_ctr):
                        src_off = n * nf * i_ctr + ic * nf
                        dst_off = n * nf * i_ctr * j_ctr + (jc * i_ctr + ic) * nf
                        gctr[dst_off:dst_off + nf] = gctr[dst_off:dst_off + nf] + c * gctri[src_off:src_off + nf]

    return gctr, has_value


def primitive_loop_1e_nuclear(envs, atm, bas, env, charge_fac, nuc_id):
    """1e nuclear attraction primitive loop."""
    shls = envs.shls
    i_sh, j_sh = int(shls[0].item()), int(shls[1].item())
    i_l, j_l = envs.i_l, envs.j_l
    i_ctr, j_ctr = int(envs.x_ctr[0].item()), int(envs.x_ctr[1].item())
    i_prim = int(bas[i_sh, NPRIM_OF].item())
    j_prim = int(bas[j_sh, NPRIM_OF].item())
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor
    ri, rj = envs.ri, envs.rj
    ai_arr = env[int(bas[i_sh, PTR_EXP].item()):int(bas[i_sh, PTR_EXP].item()) + i_prim]
    aj_arr = env[int(bas[j_sh, PTR_EXP].item()):int(bas[j_sh, PTR_EXP].item()) + j_prim]
    ci = env[int(bas[i_sh, PTR_COEFF].item()):int(bas[i_sh, PTR_COEFF].item()) + i_prim * i_ctr]
    cj = env[int(bas[j_sh, PTR_COEFF].item()):int(bas[j_sh, PTR_COEFF].item()) + j_prim * j_ctr]

    idx = compute_g_index_1e(envs)

    if nuc_id < 0:
        PTR_RINV_ORIG = 4
        cr = env[PTR_RINV_ORIG:PTR_RINV_ORIG + 3]
    else:
        cr = env[int(atm[nuc_id, PTR_COORD].item()):int(atm[nuc_id, PTR_COORD].item()) + 3]

    rrij = ((ri - rj) ** 2).sum()
    fac = charge_fac * envs.common_factor * sph_harmonic_norm(i_l) * sph_harmonic_norm(j_l)
    expcutoff = envs.expcutoff

    nc = nf * i_ctr * j_ctr
    gctr = torch.zeros(nc * n_comp, dtype=torch.float64)
    has_value = False

    g_alloc = envs.g_size * 3

    for jp in range(j_prim):
        envs.aj = aj_arr[jp].item()
        gctri = torch.zeros(nf * i_ctr * n_comp, dtype=torch.float64)

        for ip in range(i_prim):
            envs.ai = ai_arr[ip].item()
            aij = ai_arr[ip] + aj_arr[jp]
            eij = (ai_arr[ip] * aj_arr[jp] / aij) * rrij
            if eij.item() > expcutoff:
                continue
            has_value = True

            rij = (ai_arr[ip] * ri + aj_arr[jp] * rj) / aij
            tau = 1.0
            x = aij * ((rij - cr) ** 2).sum() * tau * tau
            nroots = envs.nrys_roots
            u, w = rys_roots(nroots, x)

            dij = torch.exp(-eij) / aij * fac

            gout = torch.zeros(nf * n_comp, dtype=torch.float64)
            for iroot in range(nroots):
                t2 = u[iroot] / (1.0 + u[iroot]) * tau * tau
                g = torch.zeros(g_alloc, dtype=torch.float64)
                compute_g_nuclear(g, aij, rij, cr, t2, dij * w[iroot] * tau, envs)
                envs.f_gout(gout, g, idx, envs)

            for n in range(n_comp):
                block = gout[n * nf:(n + 1) * nf]
                for ic in range(i_ctr):
                    c = ci[i_prim * ic + ip]
                    if c.item() != 0:
                        offset = n * nf * i_ctr + ic * nf
                        gctri[offset:offset + nf] = gctri[offset:offset + nf] + c * block

        for n in range(n_comp):
            for jc in range(j_ctr):
                c = cj[j_prim * jc + jp]
                if c.item() != 0:
                    for ic in range(i_ctr):
                        src_off = n * nf * i_ctr + ic * nf
                        dst_off = n * nf * i_ctr * j_ctr + (jc * i_ctr + ic) * nf
                        gctr[dst_off:dst_off + nf] = gctr[dst_off:dst_off + nf] + c * gctri[src_off:src_off + nf]

    return gctr, has_value


# 1e integral driver

INT1E_TYPE_OVLP = 0
INT1E_TYPE_RINV = 1
INT1E_TYPE_NUC = 2


def driver_1e(envs, atm, bas, env, int1e_type):
    """1e integral driver. Returns contracted spherical integrals."""
    i_ctr, j_ctr = int(envs.x_ctr[0].item()), int(envs.x_ctr[1].item())
    nf = envs.nf
    nfi, nfj = envs.nfi, envs.nfj
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor
    nc = nf * i_ctr * j_ctr

    gctr = torch.zeros(nc * n_comp, dtype=torch.float64)

    if int1e_type == INT1E_TYPE_OVLP:
        gctr, has_value = primitive_loop_1e(envs, atm, bas, env)
    elif int1e_type == INT1E_TYPE_NUC:
        natm = envs.natm
        has_value = False
        for n in range(natm):
            charge = abs(int(atm[n, 0].item()))
            if charge != 0:
                gc_n, hv = primitive_loop_1e_nuclear(envs, atm, bas, env, -charge, n)
                gctr = gctr + gc_n
                has_value = has_value or hv
    elif int1e_type == INT1E_TYPE_RINV:
        gctr, has_value = primitive_loop_1e_nuclear(envs, atm, bas, env, 1.0, -1)

    return cart_to_sph_1e(gctr, envs.i_l, envs.j_l, i_ctr, j_ctr, nfi, nfj, nf)


# Top-level 1e integral functions

def _compute_1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env, ng, f_gout, int1e_type, fac=1.0):
    """Common driver for all 1e spherical integrals."""
    envs = CINTEnvVars()
    init_envvars_1e(envs, ng, shls, atm, natm, bas, nbas, env)
    envs.f_gout = f_gout
    envs.common_factor *= fac
    result = driver_1e(envs, atm, bas, env, int1e_type)
    di = 2 * envs.i_l + 1
    dj = 2 * envs.j_l + 1
    i_ctr = int(envs.x_ctr[0].item())
    j_ctr = int(envs.x_ctr[1].item())
    naoi = dims[0]
    for jc in range(j_ctr):
        for ic in range(i_ctr):
            for j in range(dj):
                for i in range(di):
                    out[(jc * dj + j) * naoi + (ic * di + i)] = \
                        result[ic * di + i, jc * dj + j]
    return 1


def compute_overlap_1e_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure PyTorch compute_overlap_1e_sph."""
    ng = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
    return _compute_1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env,
                             ng, extract_gout_overlap, INT1E_TYPE_OVLP)


def compute_kinetic_1e_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure PyTorch compute_kinetic_1e_sph."""
    ng = torch.tensor([0, 2, 0, 0, 2, 1, 1, 1], dtype=torch.int32)
    return _compute_1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env,
                             ng, extract_gout_kinetic, INT1E_TYPE_OVLP, fac=0.5)


def compute_nuclear_1e_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure PyTorch compute_nuclear_1e_sph."""
    ng = torch.tensor([0, 0, 0, 0, 0, 1, 0, 1], dtype=torch.int32)
    return _compute_1e_sph_common(out, dims, shls, atm, natm, bas, nbas, env,
                             ng, extract_gout_nuclear, INT1E_TYPE_NUC)


# 2e integral: g-value generation

def g_rys_2d_recurrence(g, bc, envs):
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

    c00 = bc['c00']
    c0p = bc['c0p']
    b00 = bc['b00']
    b10 = bc['b10']
    b01 = bc['b01']

    for i in range(nroots):
        gx[i] = 1.0
        gy[i] = 1.0

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

    if nmax > 0 and mmax > 0:
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

        for m in range(1, mmax + 1):
            for n in range(1, nmax):
                off = m * dm + n * dn
                for i in range(nroots):
                    j = off + i
                    gx[j + dn] = c00[i, 0] * gx[j] + n * b10[i] * gx[j - dn] + m * b00[i] * gx[j - dm]
                    gy[j + dn] = c00[i, 1] * gy[j] + n * b10[i] * gy[j - dn] + m * b00[i] * gy[j - dm]
                    gz[j + dn] = c00[i, 2] * gz[j] + n * b10[i] * gz[j - dn] + m * b00[i] * gz[j - dm]


def _g_hrr_phase(gx, gy, gz, r, l_tgt, n_max, d_tgt, d_src, d_oth, l_oth, stride):
    """One phase of horizontal recurrence for 4D g-values."""
    for a in range(1, l_tgt + 1):
        for b in range(n_max - a + 1):
            for c in range(l_oth + 1):
                ptr = a * d_tgt + b * d_src + c * d_oth
                for n in range(ptr, ptr + stride):
                    gx[n] = r[0] * gx[n - d_tgt] + gx[n - d_tgt + d_src]
                    gy[n] = r[1] * gy[n - d_tgt] + gy[n - d_tgt + d_src]
                    gz[n] = r[2] * gz[n - d_tgt] + gz[n - d_tgt + d_src]


def _g_2d_to_4d_recurrence(g, envs, ij_args, kl_args):
    """Unified 4D horizontal recurrence with two phases."""
    g_size = envs.g_size
    gx, gy, gz = g[:g_size], g[g_size:2 * g_size], g[2 * g_size:3 * g_size]
    for l_tgt, n_max, d_tgt, d_src, d_oth, l_oth, stride, r in (ij_args, kl_args):
        _g_hrr_phase(gx, gy, gz, r, l_tgt, n_max, d_tgt, d_src, d_oth, l_oth, stride)


def compute_g_2e(g, fac, envs):
    """Compute 2e g-values: Rys roots/weights, then 2D recurrence, then 4D recurrence."""
    aij = envs.aij
    akl = envs.akl
    g_size = envs.g_size

    rijrkl = envs.rij - envs.rkl
    a1 = aij * akl
    a0 = a1 / (aij + akl)

    x = a0 * (rijrkl ** 2).sum()
    nroots = envs.nrys_roots

    u, w = rys_roots(nroots, x)

    fac1 = torch.sqrt(a0 / (a1 * a1 * a1)) * fac

    gz = g[2 * g_size:3 * g_size]

    if g_size == 1:
        g[0] = 1.0
        g[g_size] = 1.0
        gz[0] = w[0] * fac1
        return True

    rijrx = envs.rijrx
    rklrx = envs.rklrx

    c00 = torch.zeros((nroots, 3), dtype=torch.float64)
    c0p = torch.zeros((nroots, 3), dtype=torch.float64)
    b00 = torch.zeros(nroots, dtype=torch.float64)
    b10 = torch.zeros(nroots, dtype=torch.float64)
    b01 = torch.zeros(nroots, dtype=torch.float64)

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

    g_rys_2d_recurrence(g, bc, envs)

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
    _g_2d_to_4d_recurrence(g, envs, ij_args, kl_args)

    return True


# 2e integral loop

def primitive_loop_2e(envs, atm, bas, env):
    """2e integral primitive loop. Returns (gctr, has_value)."""
    shls = envs.shls
    i_sh, j_sh = int(shls[0].item()), int(shls[1].item())
    k_sh, l_sh = int(shls[2].item()), int(shls[3].item())
    i_ctr = int(envs.x_ctr[0].item())
    j_ctr = int(envs.x_ctr[1].item())
    k_ctr = int(envs.x_ctr[2].item())
    l_ctr = int(envs.x_ctr[3].item())
    i_prim = int(bas[i_sh, NPRIM_OF].item())
    j_prim = int(bas[j_sh, NPRIM_OF].item())
    k_prim = int(bas[k_sh, NPRIM_OF].item())
    l_prim = int(bas[l_sh, NPRIM_OF].item())

    ri, rj = envs.ri, envs.rj
    rk, rl = envs.rk, envs.rl
    ai_arr = env[int(bas[i_sh, PTR_EXP].item()):int(bas[i_sh, PTR_EXP].item()) + i_prim]
    aj_arr = env[int(bas[j_sh, PTR_EXP].item()):int(bas[j_sh, PTR_EXP].item()) + j_prim]
    ak_arr = env[int(bas[k_sh, PTR_EXP].item()):int(bas[k_sh, PTR_EXP].item()) + k_prim]
    al_arr = env[int(bas[l_sh, PTR_EXP].item()):int(bas[l_sh, PTR_EXP].item()) + l_prim]
    ci = env[int(bas[i_sh, PTR_COEFF].item()):int(bas[i_sh, PTR_COEFF].item()) + i_prim * i_ctr]
    cj = env[int(bas[j_sh, PTR_COEFF].item()):int(bas[j_sh, PTR_COEFF].item()) + j_prim * j_ctr]
    ck = env[int(bas[k_sh, PTR_COEFF].item()):int(bas[k_sh, PTR_COEFF].item()) + k_prim * k_ctr]
    cl = env[int(bas[l_sh, PTR_COEFF].item()):int(bas[l_sh, PTR_COEFF].item()) + l_prim * l_ctr]

    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_e2 * envs.ncomp_tensor
    nc = i_ctr * j_ctr * k_ctr * l_ctr

    idx = compute_g_index_2e(envs)

    expcutoff = envs.expcutoff
    dist_ij = ((ri - rj) ** 2).sum()
    dist_kl = ((rk - rl) ** 2).sum()

    gctr = torch.zeros(nf * nc * n_comp, dtype=torch.float64)
    has_value = False
    g_alloc = envs.g_size * 3

    for lp in range(l_prim):
        envs.al = al_arr[lp].item()
        gctrk = torch.zeros(nf * i_ctr * j_ctr * k_ctr * n_comp, dtype=torch.float64)

        for kp in range(k_prim):
            akl = ak_arr[kp] + al_arr[lp]
            ekl = dist_kl * ak_arr[kp] * al_arr[lp] / akl
            if ekl.item() > expcutoff:
                continue
            envs.ak = ak_arr[kp].item()
            envs.akl = akl
            envs.rkl = (ak_arr[kp] * rk + al_arr[lp] * rl) / akl
            envs.rklrx = envs.rkl - envs.rx_in_rklrx
            ekl_exp = torch.exp(-ekl)

            gctrj = torch.zeros(nf * i_ctr * j_ctr * n_comp, dtype=torch.float64)

            for jp in range(j_prim):
                envs.aj = aj_arr[jp].item()
                gctri = torch.zeros(nf * i_ctr * n_comp, dtype=torch.float64)

                for ip in range(i_prim):
                    envs.ai = ai_arr[ip].item()
                    aij = ai_arr[ip] + aj_arr[jp]
                    eij = dist_ij * ai_arr[ip] * aj_arr[jp] / aij
                    if eij.item() > expcutoff:
                        continue
                    envs.aij = aij
                    envs.rij = (ai_arr[ip] * ri + aj_arr[jp] * rj) / aij
                    envs.rijrx = envs.rij - envs.rx_in_rijrx

                    expijkl = torch.exp(-eij) * ekl_exp
                    fac1i = envs.common_factor * expijkl
                    if i_ctr == 1:
                        fac1i = fac1i * ci[ip]
                    if j_ctr == 1:
                        fac1i = fac1i * cj[jp]
                    if k_ctr == 1:
                        fac1i = fac1i * ck[kp]
                    if l_ctr == 1:
                        fac1i = fac1i * cl[lp]

                    g = torch.zeros(g_alloc, dtype=torch.float64)
                    if compute_g_2e(g, fac1i, envs):
                        has_value = True
                        gout = torch.zeros(nf * n_comp, dtype=torch.float64)
                        extract_gout_2e(gout, g, idx, envs, True)

                        if i_ctr == 1:
                            gctri[:nf * n_comp] = gctri[:nf * n_comp] + gout
                        else:
                            for ic in range(i_ctr):
                                c = ci[i_prim * ic + ip]
                                if c.item() != 0:
                                    gctri[ic * nf:(ic + 1) * nf] = gctri[ic * nf:(ic + 1) * nf] + c * gout[:nf]

                if j_ctr == 1:
                    gctrj[:nf * i_ctr * n_comp] = gctrj[:nf * i_ctr * n_comp] + gctri
                else:
                    for jc in range(j_ctr):
                        c = cj[j_prim * jc + jp]
                        if c.item() != 0:
                            sz = nf * i_ctr
                            gctrj[jc * sz:(jc + 1) * sz] = gctrj[jc * sz:(jc + 1) * sz] + c * gctri[:sz]

            if k_ctr == 1:
                gctrk[:nf * i_ctr * j_ctr * n_comp] = gctrk[:nf * i_ctr * j_ctr * n_comp] + gctrj
            else:
                for kc in range(k_ctr):
                    c = ck[k_prim * kc + kp]
                    if c.item() != 0:
                        sz = nf * i_ctr * j_ctr
                        gctrk[kc * sz:(kc + 1) * sz] = gctrk[kc * sz:(kc + 1) * sz] + c * gctrj[:sz]

        if l_ctr == 1:
            gctr[:nf * nc * n_comp] = gctr[:nf * nc * n_comp] + gctrk
        else:
            for lc in range(l_ctr):
                c = cl[l_prim * lc + lp]
                if c.item() != 0:
                    sz = nf * i_ctr * j_ctr * k_ctr
                    gctr[lc * sz:(lc + 1) * sz] = gctr[lc * sz:(lc + 1) * sz] + c * gctrk[:sz]

    return gctr, has_value


def driver_2e_sph(envs, atm, bas, env):
    """2e integral driver. Returns contracted spherical integrals."""
    x_ctr = envs.x_ctr
    i_ctr, j_ctr, k_ctr, l_ctr = [int(x.item()) for x in x_ctr]
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_e2 * envs.ncomp_tensor

    gctr, has_value = primitive_loop_2e(envs, atm, bas, env)

    if not has_value:
        di = (2 * envs.i_l + 1) * i_ctr
        dj = (2 * envs.j_l + 1) * j_ctr
        dk = (2 * envs.k_l + 1) * k_ctr
        dl = (2 * envs.l_l + 1) * l_ctr
        return torch.zeros((di, dj, dk, dl), dtype=torch.float64)

    nfi = envs.nfi
    nfj = envs.nfj
    nfk = envs.nfk
    nfl = envs.nfl
    return cart_to_sph_2e(gctr, envs.i_l, envs.j_l, envs.k_l, envs.l_l,
                          [i_ctr, j_ctr, k_ctr, l_ctr],
                          nfi, nfj, nfk, nfl, nf)


def compute_eri_2e_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure PyTorch compute_eri_2e_sph."""
    ng = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
    envs = CINTEnvVars()
    init_envvars_2e(envs, ng, shls, atm, natm, bas, nbas, env)
    envs.f_gout = extract_gout_2e
    result = driver_2e_sph(envs, atm, bas, env)

    di_total = result.shape[0]
    dj_total = result.shape[1]
    dk_total = result.shape[2]
    dl_total = result.shape[3]

    if dims is None:
        block = result.ravel()
        out[:len(block)] = block
    else:
        naoi, naoj, naok, naol = dims[0], dims[1], dims[2], dims[3]
        for l in range(dl_total):
            for k in range(dk_total):
                for j in range(dj_total):
                    for i in range(di_total):
                        flat_idx = ((l * naok + k) * naoj + j) * naoi + i
                        out[flat_idx] = result[i, j, k, l]
    return 1


# GTO driver functions

def assemble_2center_integrals(intor, out, comp, hermi, shls_slice, ao_loc,
                               opt, atm, natm, bas, nbas, env):
    """Pure PyTorch replacement for CGTO().assemble_2center_integrals."""
    ish0, ish1, jsh0, jsh1 = shls_slice[:4]
    naoi = ao_loc[ish1] - ao_loc[ish0]
    naoj = ao_loc[jsh1] - ao_loc[jsh0]

    for ish in range(ish0, ish1):
        for jsh in range(jsh0, jsh1):
            if hermi != 0 and ish > jsh:
                continue
            shls = torch.tensor([ish, jsh, 0, 0], dtype=torch.int32)
            i0 = ao_loc[ish] - ao_loc[ish0]
            j0 = ao_loc[jsh] - ao_loc[jsh0]
            dims = [naoi, naoj]
            intor(out[j0 * naoi + i0:], dims, shls,
                  atm, atm.shape[0], bas, bas.shape[0], env, opt, None)

    if hermi != 0:
        out_mat = out.reshape(naoj, naoi)
        for i in range(naoi):
            for j in range(i + 1, naoj):
                out_mat[i, j] = out_mat[j, i]


def fill_4center_s1(intor, eri, comp, ish_rel, jsh_rel, shls_slice,
                    ao_loc, opt, atm, natm, bas, nbas, env):
    """Fill function for 4-center integrals with s1 symmetry."""
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

            shls = torch.tensor([ish, jsh, ksh, lsh], dtype=torch.int32)
            buf = torch.zeros(di * dj * dk * dl, dtype=torch.float64)
            intor(buf, None, shls, atm, atm.shape[0], bas, bas.shape[0], env, opt, None)

            eri_4d[i0:i0+di, j0:j0+dj, k0:k0+dk, l0:l0+dl] = \
                buf.reshape(di, dj, dk, dl)


def fill_4center_driver(intor, fill, eri, comp, shls_slice, ao_loc,
                        opt, atm, natm, bas, nbas, env):
    """Pure PyTorch replacement for CGTO().fill_4center_driver."""
    ish0, ish1, jsh0, jsh1 = shls_slice[:4]
    nish = ish1 - ish0
    njsh = jsh1 - jsh0

    for ij in range(nish * njsh):
        i = ij // njsh
        j = ij % njsh
        fill(intor, eri, comp, i, j, shls_slice,
             ao_loc, opt, atm, natm, bas, nbas, env)


# 3-center 2-electron integrals

def primitive_loop_3c2e(envs, atm, bas, env):
    """3c2e integral primitive loop. Returns (gctr, has_value)."""
    shls = envs.shls
    i_sh, j_sh, k_sh = int(shls[0].item()), int(shls[1].item()), int(shls[2].item())
    i_ctr = int(envs.x_ctr[0].item())
    j_ctr = int(envs.x_ctr[1].item())
    k_ctr = int(envs.x_ctr[2].item())
    i_prim = int(bas[i_sh, NPRIM_OF].item())
    j_prim = int(bas[j_sh, NPRIM_OF].item())
    k_prim = int(bas[k_sh, NPRIM_OF].item())

    ri, rj = envs.ri, envs.rj
    ai_arr = env[int(bas[i_sh, PTR_EXP].item()):int(bas[i_sh, PTR_EXP].item()) + i_prim]
    aj_arr = env[int(bas[j_sh, PTR_EXP].item()):int(bas[j_sh, PTR_EXP].item()) + j_prim]
    ak_arr = env[int(bas[k_sh, PTR_EXP].item()):int(bas[k_sh, PTR_EXP].item()) + k_prim]
    ci = env[int(bas[i_sh, PTR_COEFF].item()):int(bas[i_sh, PTR_COEFF].item()) + i_prim * i_ctr]
    cj = env[int(bas[j_sh, PTR_COEFF].item()):int(bas[j_sh, PTR_COEFF].item()) + j_prim * j_ctr]
    ck = env[int(bas[k_sh, PTR_COEFF].item()):int(bas[k_sh, PTR_COEFF].item()) + k_prim * k_ctr]

    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor
    nc = i_ctr * j_ctr * k_ctr

    idx = compute_g_index_2e(envs)

    expcutoff = envs.expcutoff
    dist_ij = ((ri - rj) ** 2).sum()

    gctr = torch.zeros(nf * nc * n_comp, dtype=torch.float64)
    has_value = False
    g_alloc = envs.g_size * 3

    for kp in range(k_prim):
        envs.ak = ak_arr[kp].item()
        envs.akl = ak_arr[kp]

        gctrj = torch.zeros(nf * i_ctr * j_ctr * n_comp, dtype=torch.float64)

        for jp in range(j_prim):
            envs.aj = aj_arr[jp].item()
            gctri = torch.zeros(nf * i_ctr * n_comp, dtype=torch.float64)

            for ip in range(i_prim):
                envs.ai = ai_arr[ip].item()
                aij = ai_arr[ip] + aj_arr[jp]
                eij = dist_ij * ai_arr[ip] * aj_arr[jp] / aij
                if eij.item() > expcutoff:
                    continue
                envs.aij = aij
                envs.rij = (ai_arr[ip] * ri + aj_arr[jp] * rj) / aij
                envs.rijrx = envs.rij - envs.rx_in_rijrx

                expij = torch.exp(-eij)
                fac1i = envs.common_factor * expij
                if i_ctr == 1:
                    fac1i = fac1i * ci[ip]
                if j_ctr == 1:
                    fac1i = fac1i * cj[jp]
                if k_ctr == 1:
                    fac1i = fac1i * ck[kp]

                g = torch.zeros(g_alloc, dtype=torch.float64)
                if compute_g_2e(g, fac1i, envs):
                    has_value = True
                    gout = torch.zeros(nf * n_comp, dtype=torch.float64)
                    extract_gout_2e(gout, g, idx, envs, True)

                    if i_ctr == 1:
                        gctri[:nf * n_comp] = gctri[:nf * n_comp] + gout
                    else:
                        for ic in range(i_ctr):
                            c = ci[i_prim * ic + ip]
                            if c.item() != 0:
                                gctri[ic * nf:(ic + 1) * nf] = gctri[ic * nf:(ic + 1) * nf] + c * gout[:nf]

            if j_ctr == 1:
                gctrj[:nf * i_ctr * n_comp] = gctrj[:nf * i_ctr * n_comp] + gctri
            else:
                for jc in range(j_ctr):
                    c = cj[j_prim * jc + jp]
                    if c.item() != 0:
                        off_j = jc * nf * i_ctr
                        gctrj[off_j:off_j + nf * i_ctr] = gctrj[off_j:off_j + nf * i_ctr] + c * gctri[:nf * i_ctr]

        if k_ctr == 1:
            gctr[:nf * nc * n_comp] = gctr[:nf * nc * n_comp] + gctrj
        else:
            for kc in range(k_ctr):
                c = ck[k_prim * kc + kp]
                if c.item() != 0:
                    off_k = kc * nf * i_ctr * j_ctr
                    gctrj_size = nf * i_ctr * j_ctr
                    gctr[off_k:off_k + gctrj_size] = gctr[off_k:off_k + gctrj_size] + c * gctrj[:gctrj_size]

    return gctr, has_value


def cart_to_sph_3c2e(gctr, i_l, j_l, k_l, x_ctr, nfi, nfj, nfk, nf):
    """Cart-to-spherical transformation for 3c2e integrals."""
    i_ctr, j_ctr, k_ctr = x_ctr[0], x_ctr[1], x_ctr[2]
    di = 2 * i_l + 1
    dj = 2 * j_l + 1
    dk = 2 * k_l + 1

    c2s_i = cart_to_sph_matrix(i_l)
    c2s_j = cart_to_sph_matrix(j_l)
    c2s_k = cart_to_sph_matrix(k_l)

    nc = i_ctr * j_ctr * k_ctr
    gctr_3d = gctr.reshape(nc, nf)

    out = torch.zeros((di * i_ctr, dj * j_ctr, dk * k_ctr), dtype=torch.float64)

    n = 0
    for kc in range(k_ctr):
        for jc in range(j_ctr):
            for ic in range(i_ctr):
                block = gctr_3d[n].reshape(nfj, nfk, nfi)
                tmp = torch.tensordot(c2s_j, block, dims=([1], [0]))
                tmp = torch.tensordot(tmp, c2s_k.T, dims=([1], [0]))
                tmp = torch.tensordot(tmp, c2s_i.T, dims=([2], [0]))
                out[ic*di:(ic+1)*di, jc*dj:(jc+1)*dj, kc*dk:(kc+1)*dk] = tmp.permute(2, 0, 1)
                n += 1

    return out


def driver_3c2e_sph(envs, atm, bas, env):
    """3c2e integral driver. Returns contracted spherical integrals."""
    x_ctr = envs.x_ctr
    i_ctr, j_ctr, k_ctr = int(x_ctr[0].item()), int(x_ctr[1].item()), int(x_ctr[2].item())
    nf = envs.nf
    n_comp = envs.ncomp_e1 * envs.ncomp_tensor

    gctr, has_value = primitive_loop_3c2e(envs, atm, bas, env)

    if not has_value:
        di = (2 * envs.i_l + 1) * i_ctr
        dj = (2 * envs.j_l + 1) * j_ctr
        dk = (2 * envs.k_l + 1) * k_ctr
        return torch.zeros((di, dj, dk), dtype=torch.float64)

    nfi = envs.nfi
    nfj = envs.nfj
    nfk = envs.nfk
    return cart_to_sph_3c2e(gctr, envs.i_l, envs.j_l, envs.k_l,
                            [i_ctr, j_ctr, k_ctr], nfi, nfj, nfk, nf)


def compute_eri_3c2e_sph(out, dims, shls, atm, natm, bas, nbas, env, opt=None, cache=None):
    """Pure PyTorch compute_eri_3c2e_sph."""
    ng = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
    envs = CINTEnvVars()
    init_envvars_3c2e(envs, ng, shls, atm, natm, bas, nbas, env)
    envs.f_gout = extract_gout_2e
    result = driver_3c2e_sph(envs, atm, bas, env)

    if dims is None:
        block = result.ravel()
        out[:len(block)] = block
    else:
        naoi, naoj, naok = dims[0], dims[1], dims[2]
        di_total, dj_total, dk_total = result.shape
        for k in range(dk_total):
            for j in range(dj_total):
                for i in range(di_total):
                    flat_idx = (k * naoj + j) * naoi + i
                    out[flat_idx] = result[i, j, k]
    return 1


def fill_3center_s1(intor, out, buf, comp, jobid, shls_slice, ao_loc,
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
            shls = torch.tensor([ish, jsh, ksh, 0], dtype=torch.int32)
            i0 = ao_loc[ish] - ao_loc[ish0]
            j0 = ao_loc[jsh] - ao_loc[jsh0]
            intor(out[out_offset + j0 * naoi + i0:], dims, shls,
                  atm, natm, bas, nbas, env, opt, None)


def fill_3center_driver(intor, fill, eri, comp, shls_slice, ao_loc,
                        opt, atm, natm, bas, nbas, env):
    """Pure PyTorch replacement for CGTO().fill_3center_driver."""
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


# Fourier Transform of GTO basis functions

def _ft_1d_polynomial(k, n, a2):
    """Compute the 1D polynomial P_n(-ik*a2) for FT of x^n * exp(-alpha*x^2)."""
    if n == 0:
        return torch.ones(len(k), dtype=torch.complex128)
    ikha = -1j * k * a2
    if n == 1:
        return ikha
    p_prev = torch.ones(len(k), dtype=torch.complex128)
    p_curr = ikha.clone()
    for m in range(1, n):
        p_next = m * a2 * p_prev + ikha * p_curr
        p_prev = p_curr
        p_curr = p_next
    return p_curr


def evaluate_gto_ft(wrapper, gvgrid):
    """Pure PyTorch FT evaluator for GTO basis functions."""
    from deepchem.utils.analytical_integrators_torch.optimizer import (
        sph_harmonic_norm, ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF,
        PTR_EXP, PTR_COEFF, PTR_COORD,
    )

    _atm, _bas, _env = wrapper.atm_bas_env
    atm = torch.as_tensor(_atm, dtype=torch.int32)
    bas = torch.as_tensor(_bas, dtype=torch.int32)
    env = torch.as_tensor(_env, dtype=torch.float64)
    ao_loc = wrapper.full_shell_to_aoloc
    nao = wrapper.nao()
    Gv = gvgrid.to(torch.float64)
    NGv = Gv.shape[0]
    Gx, Gy, Gz = Gv[:, 0], Gv[:, 1], Gv[:, 2]
    G2 = Gx**2 + Gy**2 + Gz**2

    out = torch.zeros((nao, NGv), dtype=torch.complex128)

    ish0, ish1 = wrapper.shell_idxs
    for ish in range(ish0, ish1):
        l = int(bas[ish, ANG_OF].item())
        nprim = int(bas[ish, NPRIM_OF].item())
        nctr = int(bas[ish, NCTR_OF].item())
        nf_cart = (l + 1) * (l + 2) // 2
        di = 2 * l + 1

        atom_idx = int(bas[ish, ATOM_OF].item())
        R = env[int(atm[atom_idx, PTR_COORD].item()):int(atm[atom_idx, PTR_COORD].item()) + 3]
        alphas = env[int(bas[ish, PTR_EXP].item()):int(bas[ish, PTR_EXP].item()) + nprim]
        coeffs = env[int(bas[ish, PTR_COEFF].item()):int(bas[ish, PTR_COEFF].item()) + nprim * nctr]

        phase = torch.exp(-1j * (Gx * R[0] + Gy * R[1] + Gz * R[2]))

        i_nx, i_ny, i_nz = cartesian_components(l)

        fac_norm = (math.sqrt(math.pi) * math.pi
                    * sph_harmonic_norm(l) * sph_harmonic_norm(0)
                    * math.sqrt(4.0 * math.pi))

        for ic in range(nctr):
            cart_ft = torch.zeros((nf_cart, NGv), dtype=torch.complex128)

            for ip in range(nprim):
                alpha = alphas[ip]
                c = coeffs[ic * nprim + ip]
                if abs(c.item()) < 1e-30:
                    continue

                a2 = 0.5 / alpha
                radial = fac_norm * c / (alpha * torch.sqrt(alpha))
                exp_factor = torch.exp(-G2 * a2 * 0.5)
                base = radial * exp_factor * phase

                max_n = l
                px = {}
                py = {}
                pz = {}
                for n in range(max_n + 1):
                    px[n] = _ft_1d_polynomial(Gx, n, a2)
                    py[n] = _ft_1d_polynomial(Gy, n, a2)
                    pz[n] = _ft_1d_polynomial(Gz, n, a2)

                for f in range(nf_cart):
                    a, b, cc = int(i_nx[f].item()), int(i_ny[f].item()), int(i_nz[f].item())
                    cart_ft[f] = cart_ft[f] + base * px[a] * py[b] * pz[cc]

            c2s = cart_to_sph_matrix(l).to(torch.complex128)
            sph_ft = c2s @ cart_ft

            ao_start = ao_loc[ish] - ao_loc[ish0] + ic * di
            out[ao_start:ao_start + di, :] = sph_ft

    return out


# GTO grid evaluator

def evaluate_gto_grid(wrapper, shortname, rgrid, spherical):
    """Pure PyTorch GTO grid evaluator."""
    import re

    _atm, _bas, _env = wrapper.atm_bas_env
    atm = torch.as_tensor(_atm, dtype=torch.int32)
    bas = torch.as_tensor(_bas, dtype=torch.int32)
    env = torch.as_tensor(_env, dtype=torch.float64)
    ao_loc = wrapper.full_shell_to_aoloc
    nao = wrapper.nao()
    coords = rgrid.to(torch.float64)
    ngrid = coords.shape[0]

    n_ip = len(re.findall(r"^(?:ip)*(?:ip)?", shortname)[0]) // 2
    comp_shape = (3,) * n_ip
    ncomp = 3 ** n_ip if n_ip > 0 else 1

    if ncomp > 1:
        out = torch.zeros(comp_shape + (nao, ngrid), dtype=torch.float64)
    else:
        out = torch.zeros((nao, ngrid), dtype=torch.float64)

    ish0, ish1 = wrapper.shell_idxs

    for ish in range(ish0, ish1):
        l = int(bas[ish, ANG_OF].item())
        nprim = int(bas[ish, NPRIM_OF].item())
        nctr = int(bas[ish, NCTR_OF].item())
        atom_idx = int(bas[ish, ATOM_OF].item())
        nf_cart = (l + 1) * (l + 2) // 2
        di = 2 * l + 1 if spherical else nf_cart

        R = env[int(atm[atom_idx, PTR_COORD].item()):int(atm[atom_idx, PTR_COORD].item()) + 3]
        alphas_arr = env[int(bas[ish, PTR_EXP].item()):int(bas[ish, PTR_EXP].item()) + nprim]
        coeffs_flat = env[int(bas[ish, PTR_COEFF].item()):int(bas[ish, PTR_COEFF].item()) + nprim * nctr]

        fac = sph_harmonic_norm(l)

        rx = coords[:, 0] - R[0]
        ry = coords[:, 1] - R[1]
        rz = coords[:, 2] - R[2]
        r2 = rx * rx + ry * ry + rz * rz

        eprim = torch.exp(-alphas_arr[:, None] * r2[None, :]) * fac

        max_pow = l
        if 'ip' in shortname:
            max_pow = max(max_pow, l + 1)
        if 'lapl' in shortname or 'rr' in shortname:
            max_pow = max(max_pow, l + 2)

        xpows = torch.ones((max_pow + 1, ngrid), dtype=torch.float64)
        ypows = torch.ones((max_pow + 1, ngrid), dtype=torch.float64)
        zpows = torch.ones((max_pow + 1, ngrid), dtype=torch.float64)
        for p in range(1, max_pow + 1):
            xpows[p] = xpows[p - 1] * rx
            ypows[p] = ypows[p - 1] * ry
            zpows[p] = zpows[p - 1] * rz

        cart_indices = []
        for lx in range(l, -1, -1):
            for ly in range(l - lx, -1, -1):
                lz = l - lx - ly
                cart_indices.append((lx, ly, lz))

        for ic in range(nctr):
            coeff_ic = coeffs_flat[ic * nprim:(ic + 1) * nprim]

            if shortname == "":
                ectr = coeff_ic @ eprim
                cart_vals = torch.empty((nf_cart, ngrid), dtype=torch.float64)
                for f, (lx, ly, lz) in enumerate(cart_indices):
                    cart_vals[f] = ectr * xpows[lx] * ypows[ly] * zpows[lz]

            elif shortname == "ip":
                ectr = coeff_ic @ eprim
                ectr_2a = ((-2 * alphas_arr) * coeff_ic) @ eprim
                cart_vals = torch.zeros((3, nf_cart, ngrid), dtype=torch.float64)
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
                cart_vals = torch.zeros((nf_cart, ngrid), dtype=torch.float64)
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
                cart_vals = torch.empty((nf_cart, ngrid), dtype=torch.float64)
                for f, (lx, ly, lz) in enumerate(cart_indices):
                    cart_vals[f] = (ectr * r2
                                    * xpows[lx] * ypows[ly] * zpows[lz])

            else:
                raise NotImplementedError(
                    "GTO grid evaluation for shortname='%s' not implemented"
                    % shortname)

            if spherical and l >= 2:
                c2s = cart_to_sph_matrix(l)
                if ncomp > 1:
                    sph = torch.empty(comp_shape + (di, ngrid), dtype=torch.float64)
                    for idx_t in torch.cartesian_prod(*[torch.arange(s) for s in comp_shape]):
                        idx_tuple = tuple(idx_t.tolist()) if idx_t.dim() > 0 else (idx_t.item(),)
                        sph[idx_tuple] = c2s @ cart_vals[idx_tuple]
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


# Integral function registry

INTEGRAL_REGISTRY = {
    'int1e_ovlp_sph': compute_overlap_1e_sph,
    'int1e_kin_sph': compute_kinetic_1e_sph,
    'int1e_nuc_sph': compute_nuclear_1e_sph,
    'int2e_ar12b_sph': compute_eri_2e_sph,
    'int3c2e_ar12_sph': compute_eri_3c2e_sph,
}
