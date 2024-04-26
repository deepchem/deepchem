from __future__ import annotations
from typing import Mapping, Tuple, Optional, Union, Iterator, List
import torch
import numpy as np
import warnings
try:
    import pylibxc
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(
        f"{e}, Failed to import pylibxc. Might not be able to use xc.")

# libxc with derivative

# This is the interface of libxc to pytorch to make the it differentiable
# in pytorch format.
# The torch inputs are flattened and should have been checked to have the
# same length and shape, i.e. (ninps).


class CalcLDALibXCUnpol(torch.autograd.Function):
    """Calculates the energy density or its derivative w.r.t. density for
    unpolarized LDA.

    Local-density approximations (LDA) are a class of approximations to the
    exchange–correlation (XC) energy functional in density functional theory
    (DFT) that depend solely upon the value of the electronic density at each
    point in space (and not, for example, derivatives of the density or the
    Kohn–Sham orbitals).

    The result is a tensor with shape (ninps).

    Examples
    --------
    >>> import torch
    >>> import pylibxc
    >>> libxcfcn = pylibxc.LibXCFunctional("lda_x", "unpolarized")
    >>> rho = torch.tensor([0.1, 0.2, 0.3])
    >>> res = CalcLDALibXCUnpol.apply(rho, 0, libxcfcn)[0]
    >>> print(res)
    tensor([[-0.0343, -0.0864, -0.1483]], dtype=torch.float64)

    """

    @staticmethod
    def forward(ctx, rho: torch.Tensor, deriv: int,  # type: ignore
                libxcfcn: pylibxc.functional.LibXCFunctional) -> \
            Tuple[torch.Tensor, ...]:  # type: ignore
        """Calculates and returns the energy density or its derivative w.r.t.
        density.

        Parameters
        ----------
        rho: torch.Tensor
            Density tensor with shape (ninps)
        deriv: int
            Derivative order. 0 for energy density, 1 for derivative w.r.t.
            density, 2 for second derivative w.r.t. density, etc.
        libxcfcn: pylibxc.functional.LibXCFunctional
            libxc functional to use

        Returns
        -------
        Tuple[torch.Tensor]
            Result is a tensor with shape (ninps)

        """

        inp = {
            "rho": rho.detach().numpy(),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=1, polarized=False)[0]

        ctx.save_for_backward(rho, res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (res,)

    @staticmethod
    def backward(
        ctx, *grad_res: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Calculates the gradient w.r.t. the input rho.

        Parameters
        ----------
        grad_res: torch.Tensor
            Gradient of the result w.r.t. the result itself.

        Returns
        -------
        Tuple[torch.Tensor]
            Gradient w.r.t. the input rho.

        """
        rho, res = ctx.saved_tensors

        dres_drho = CalcLDALibXCUnpol.apply(rho, ctx.deriv + 1, ctx.libxcfcn)[0]
        grad_rho = dres_drho * grad_res[0]
        return (grad_rho, None, None)


class CalcLDALibXCPol(torch.autograd.Function):
    """
    Local-density approximations (LDA) are a class of approximations to the
    exchange–correlation (XC) energy functional in density functional theory
    (DFT) that depend solely upon the value of the electronic density at each
    point in space (and not, for example, derivatives of the density or the
    Kohn–Sham orbitals).

    Examples
    --------
    >>> import torch
    >>> import pylibxc
    >>> libxcfcn = pylibxc.LibXCFunctional("lda_x", "polarized")
    >>> rho_u = torch.tensor([0.1, 0.2, 0.3])
    >>> rho_d = torch.tensor([0.1, 0.2, 0.3])
    >>> res = CalcLDALibXCPol.apply(rho_u, rho_d, 0, libxcfcn)[0]
    >>> print(res)
    tensor([[-0.0864, -0.2177, -0.3738]], dtype=torch.float64)

    """

    @staticmethod
    def forward(
        ctx,
        rho_u: torch.Tensor,
        rho_d: torch.Tensor,
        deriv: int,  # type: ignore
        libxcfcn: pylibxc.functional.LibXCFunctional
    ) -> Tuple[torch.Tensor, ...]:
        """Calculates and returns the energy density or its derivative w.r.t.
        density for polarized LDA.

        Parameters
        ----------
        rho_u: torch.Tensor
            Density tensor for spin-up with shape (ninps)
        rho_d: torch.Tensor
            Density tensor for spin-down with shape (ninps)
        deriv: int
            Derivative order. 0 for energy density, 1 for derivative w.r.t.
            density, 2 for second derivative w.r.t. density, etc.
        libxcfcn: pylibxc.functional.LibXCFunctional
            libxc functional to use

        Returns
        -------
        Tuple[torch.Tensor]
            Result is a tensor with shape (nderiv, ninps) where the first
            dimension indicates the result for derivatives of spin-up and
            spin-down and some of its combination.

        """

        inp = {
            "rho": _pack_input(rho_u, rho_d),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=1, polarized=True)[0]

        ctx.save_for_backward(rho_u, rho_d, res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (res,)

    @staticmethod
    def backward(ctx,  # type: ignore
                 *grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Calculates the gradient w.r.t. the input rho.

        Parameters
        ----------
        grad_res: torch.Tensor
            Gradient of the result w.r.t. the result itself.

        Returns
        -------
        Tuple[torch.Tensor]
            Gradient w.r.t. the input rho.

        """
        inps = ctx.saved_tensors[:2]
        res = ctx.saved_tensors[2:]  # noqa: F841
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        derivs = CalcLDALibXCPol.apply(*inps, deriv + 1, libxcfcn)

        # generated by `_generate_spin_list(deriv, ["rho"], [2])`
        if deriv == 0:
            deriv_idxs = [[0], [0]]
            spin_idxs: List[List[Tuple[int, ...]]] = [[(0,)], [(1,)]]
        elif deriv == 1:
            deriv_idxs = [[0], [0]]
            spin_idxs = [[(0, 1)], [(1, 2)]]
        elif deriv == 2:
            deriv_idxs = [[0], [0]]
            spin_idxs = [[(0, 1, 2)], [(1, 2, 3)]]
        elif deriv == 3:
            deriv_idxs = [[0], [0]]
            spin_idxs = [[(0, 1, 2, 3)], [(1, 2, 3, 4)]]
        else:
            raise RuntimeError(
                f"Unimplemented derivative for deriv == {deriv} for polarized LDA"
            )

        grad_inps = _get_grad_inps(grad_res, inps, derivs, ctx.needs_input_grad,
                                   deriv_idxs, spin_idxs)
        return (*grad_inps, None, None)


class CalcGGALibXCUnpol(torch.autograd.Function):
    """Calculates the energy density or its derivative w.r.t. density for
    unpolarized GGA.

    Generalized-gradient approximations (GGA) are a class of approximations to
    the exchange–correlation (XC) energy functional in density functional theory
    (DFT) that depend not only upon the value of the electronic density at each
    point in space, but also upon its gradient.

    Examples
    --------
    >>> import torch
    >>> import pylibxc
    >>> libxcfcn = pylibxc.LibXCFunctional("gga_c_pbe", "unpolarized")
    >>> rho = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma = torch.tensor([0.1, 0.2, 0.3])
    >>> res = CalcGGALibXCUnpol.apply(rho, sigma, 0, libxcfcn)[0]
    >>> print(res)
    tensor([[-0.0016, -0.0070, -0.0137]], dtype=torch.float64)

    """

    @staticmethod
    def forward(ctx, rho: torch.Tensor, sigma: torch.Tensor, deriv: int,  # type: ignore
                libxcfcn: pylibxc.functional.LibXCFunctional) ->\
            Tuple[torch.Tensor, ...]:  # type: ignore
        """Calculates and returns the energy density or its derivative w.r.t.
        density and contracted gradient.

        Every element in the tuple is a tensor with shape (ninps)

        Parameters
        ----------
        rho: torch.Tensor
            Density tensor with shape (ninps)
        sigma: torch.Tensor
            Contracted gradient tensor with shape (ninps)
        deriv: int
            Derivative order. 0 for energy density, 1 for derivative w.r.t.
            density, 2 for second derivative w.r.t. density, etc.
        libxcfcn: pylibxc.functional.LibXCFunctional
            libxc functional to use

        """

        inp = {
            "rho": rho,
            "sigma": sigma,
        }
        # for gga, res is a tuple
        res = _get_libxc_res(inp, deriv, libxcfcn, family=2, polarized=False)

        ctx.save_for_backward(rho, sigma, *res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (*res,)

    @staticmethod
    def backward(ctx, *grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Calculates the gradient w.r.t. the input rho and sigma.

        Parameters
        ----------
        grad_res : torch.Tensor
            Gradient of the result w.r.t. the result itself.

        Returns
        -------
        Tuple[torch.Tensor]
            Gradient w.r.t. the input rho and sigma.

        """
        inps = ctx.saved_tensors[:2]
        res = ctx.saved_tensors[2:]  # noqa: F841
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        derivs = CalcGGALibXCUnpol.apply(*inps, deriv + 1, libxcfcn)

        # generated by _generate_pair_deriv_idxs(deriv, ["rho", "sigma"])
        # see _get_grad_inps for explanation about deriv_idxs
        if deriv == 0:
            deriv_idxs = [[0], [1]]
        elif deriv == 1:
            deriv_idxs = [[0, 1], [1, 2]]
        elif deriv == 2:
            deriv_idxs = [[0, 1, 2], [1, 2, 3]]
        elif deriv == 3:
            deriv_idxs = [[0, 1, 2, 3], [1, 2, 3, 4]]
        else:
            raise RuntimeError("Cannot handle GGA deriv %d" % deriv)

        grad_inps = _get_grad_inps(grad_res, inps, derivs, ctx.needs_input_grad,
                                   deriv_idxs)
        return (*grad_inps, None, None)


class CalcGGALibXCPol(torch.autograd.Function):
    """Calculates the energy density or its derivative w.r.t. density for
    polarized GGA.

    Generalized-gradient approximations (GGA) are a class of approximations to
    the exchange–correlation (XC) energy functional in density functional theory
    (DFT) that depend not only upon the value of the electronic density at each
    point in space, but also upon its gradient.

    Examples
    --------
    >>> import torch
    >>> import pylibxc
    >>> libxcfcn = pylibxc.LibXCFunctional("gga_c_pbe", "polarized")
    >>> rho_u = torch.tensor([0.1, 0.2, 0.3])
    >>> rho_d = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma_uu = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma_ud = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma_dd = torch.tensor([0.1, 0.2, 0.3])
    >>> res = CalcGGALibXCPol.apply(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, 0, libxcfcn)[0]
    >>> print(res)
    tensor([[-0.0047, -0.0175, -0.0322]], dtype=torch.float64)

    """

    @staticmethod
    def forward(ctx, rho_u: torch.Tensor, rho_d: torch.Tensor,  # type: ignore
                sigma_uu: torch.Tensor, sigma_ud: torch.Tensor, sigma_dd: torch.Tensor,
                deriv: int, libxcfcn: pylibxc.functional.LibXCFunctional) -> \
            Tuple[torch.Tensor, ...]:  # type: ignore
        """
        Calculates and returns the energy density or its derivative w.r.t.
        density and contracted gradient.

        Parameters
        ----------
        rho_u: torch.Tensor
            Density tensor for spin-up with shape (ninps)
        rho_d: torch.Tensor
            Density tensor for spin-down with shape (ninps)
        sigma_uu: torch.Tensor
            Contracted gradient tensor for spin-up with shape (ninps)
        sigma_ud: torch.Tensor
            Contracted gradient tensor for spin-up and spin-down with shape (ninps)
        sigma_dd: torch.Tensor
            Contracted gradient tensor for spin-down with shape (ninps)
        deriv: int
            Derivative order. 0 for energy density, 1 for derivative w.r.t.
            density, 2 for second derivative w.r.t. density, etc.
        libxcfcn: pylibxc.functional.LibXCFunctional
            libxc functional to use

        Returns
        -------
        Tuple[torch.Tensor]
            Result is a tensor with shape (nderiv, ninps) where the first
            dimension indicates the result for derivatives of spin-up and
            spin-down and some of its combination.

        """

        inp = {
            "rho": _pack_input(rho_u, rho_d),
            "sigma": _pack_input(sigma_uu, sigma_ud, sigma_dd),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=2, polarized=True)

        ctx.save_for_backward(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, *res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (*res,)

    @staticmethod
    def backward(ctx, *grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Returns the gradient w.r.t. the input rho and sigma.

        Parameters
        ----------
        grad_res : torch.Tensor
            Gradient of the result w.r.t. the result itself.

        Returns
        -------
        Tuple[torch.Tensor]
            Gradient w.r.t. the input rho and sigma.

        """
        inps = ctx.saved_tensors[:5]
        res = ctx.saved_tensors[5:]  # noqa: F841
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        derivs = CalcGGALibXCPol.apply(*inps, deriv + 1, libxcfcn)

        # generated by `_generate_spin_list(deriv, ["rho", "sigma"], [2, 3])`
        if deriv == 0:
            deriv_idxs = [[0], [0], [1], [1], [1]]
            spin_idxs: List[List[Tuple[int, ...]]] = [[(0,)], [(1,)], [(0,)],
                                                      [(1,)], [(2,)]]
        elif deriv == 1:
            deriv_idxs = [[0, 1], [0, 1], [1, 2], [1, 2], [1, 2]]
            spin_idxs = [[(0, 1), (0, 1, 2)], [(1, 2), (3, 4, 5)],
                         [(0, 3), (0, 1, 2)], [(1, 4), (1, 3, 4)],
                         [(2, 5), (2, 4, 5)]]
        elif deriv == 2:
            deriv_idxs = [[0, 1, 2], [0, 1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
            spin_idxs = [[(0, 1, 2), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)],
                         [(1, 2, 3), (3, 4, 5, 6, 7, 8), (6, 7, 8, 9, 10, 11)],
                         [(0, 3, 6), (0, 1, 2, 6, 7, 8), (0, 1, 2, 3, 4, 5)],
                         [(1, 4, 7), (1, 3, 4, 7, 9, 10), (1, 3, 4, 6, 7, 8)],
                         [(2, 5, 8), (2, 4, 5, 8, 10, 11), (2, 4, 5, 7, 8, 9)]]
        elif deriv == 3:
            deriv_idxs = [[0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 4],
                          [1, 2, 3, 4], [1, 2, 3, 4]]
            spin_idxs = [[(0, 1, 2, 3), (0, 1, 2, 3, 4, 5, 6, 7, 8),
                          (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
                          (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)],
                         [(1, 2, 3, 4), (3, 4, 5, 6, 7, 8, 9, 10, 11),
                          (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
                          (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)],
                         [(0, 3, 6, 9), (0, 1, 2, 6, 7, 8, 12, 13, 14),
                          (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15),
                          (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)],
                         [(1, 4, 7, 10), (1, 3, 4, 7, 9, 10, 13, 15, 16),
                          (1, 3, 4, 6, 7, 8, 11, 13, 14, 16, 17, 18),
                          (1, 3, 4, 6, 7, 8, 10, 11, 12, 13)],
                         [(2, 5, 8, 11), (2, 4, 5, 8, 10, 11, 14, 16, 17),
                          (2, 4, 5, 7, 8, 9, 12, 14, 15, 17, 18, 19),
                          (2, 4, 5, 7, 8, 9, 11, 12, 13, 14)]]
        else:
            raise RuntimeError(
                f"Unimplemented derivative for deriv == {deriv} for polarized GGA"
            )

        grad_inps = _get_grad_inps(grad_res, inps, derivs, ctx.needs_input_grad,
                                   deriv_idxs, spin_idxs)
        return (*grad_inps, None, None)


class CalcMGGALibXCUnpol(torch.autograd.Function):
    """Calculates the energy density or its derivative w.r.t. density for
    unpolarized meta-GGA.

    Meta-generalized-gradient approximations (meta-GGA) are a class of approximations
    to the exchange–correlation (XC) energy functional in density functional theory
    (DFT) that depend not only upon the value of the electronic density at each
    point in space and its gradient, but also upon the Laplacian of the density
    and the kinetic energy density.

    Examples
    --------
    >>> import torch
    >>> import pylibxc
    >>> libxcfcn = pylibxc.LibXCFunctional("mgga_c_m06_l", "unpolarized")
    >>> rho = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma = torch.tensor([0.1, 0.2, 0.3])
    >>> lapl = torch.tensor([0.1, 0.2, 0.3])
    >>> kin = torch.tensor([0.1, 0.2, 0.3])
    >>> res = CalcMGGALibXCUnpol.apply(rho, sigma, lapl, kin, 0, libxcfcn)[0]
    >>> print(res)
    tensor([[-0.0032, -0.0066, -0.0087]], dtype=torch.float64)

    """

    @staticmethod
    def forward(ctx, rho: torch.Tensor, sigma: torch.Tensor, lapl: torch.Tensor,  # type: ignore
                kin: torch.Tensor, deriv: int,
                libxcfcn: pylibxc.functional.LibXCFunctional) ->\
            Tuple[torch.Tensor, ...]:  # type: ignore
        """
        Calculates and returns the energy density or its derivative w.r.t.
        density and contracted gradient.

        Parameters
        ----------
        rho: torch.Tensor
            Density tensor with shape (ninps)
        sigma: torch.Tensor
            Contracted gradient tensor with shape (ninps)
        lapl: torch.Tensor
            Laplacian tensor with shape (ninps)
        kin: torch.Tensor
            Kinetic energy density tensor with shape (ninps)
        deriv: int
            Derivative order. 0 for energy density, 1 for derivative w.r.t.
            density, 2 for second derivative w.r.t. density, etc.
        libxcfcn: pylibxc.functional.LibXCFunctional
            libxc functional to use

        Returns
        -------
        Tuple[torch.Tensor]
            The result is a tensor with shape (nderiv, ninps)

        """

        inp = {
            "rho": rho,
            "sigma": sigma,
            "lapl": lapl,
            "tau": kin,
        }
        # res is a tuple
        res = _get_libxc_res(inp, deriv, libxcfcn, family=4, polarized=False)

        ctx.save_for_backward(rho, sigma, lapl, kin, *res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (*res,)

    @staticmethod
    def backward(ctx, *grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Returns the gradient w.r.t. the inputs

        Parameters
        ----------
        grad_res : torch.Tensor
            The gradient of the result w.r.t. the result itself.

        Returns
        -------
        Tuple[torch.Tensor]
            The gradient w.r.t. the inputs

        """
        inps = ctx.saved_tensors[:4]
        res = ctx.saved_tensors[4:]  # noqa: F841
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        derivs = CalcMGGALibXCUnpol.apply(*inps, deriv + 1, libxcfcn)

        # generated by _generate_pair_deriv_idxs(deriv, ["rho", "sigma", "lapl", "tau"])
        # see _get_grad_inps for explanation about deriv_idxs
        if deriv == 0:
            deriv_idxs = [[0], [1], [2], [3]]
        elif deriv == 1:
            deriv_idxs = [[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8],
                          [3, 6, 8, 9]]
        elif deriv == 2:
            deriv_idxs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          [1, 4, 5, 6, 10, 11, 12, 13, 14, 15],
                          [2, 5, 7, 8, 11, 13, 14, 16, 17, 18],
                          [3, 6, 8, 9, 12, 14, 15, 17, 18, 19]]
        elif deriv == 3:
            deriv_idxs = [[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19
            ],
                          [
                              1, 4, 5, 6, 10, 11, 12, 13, 14, 15, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 29
                          ],
                          [
                              2, 5, 7, 8, 11, 13, 14, 16, 17, 18, 21, 23, 24,
                              26, 27, 28, 30, 31, 32, 33
                          ],
                          [
                              3, 6, 8, 9, 12, 14, 15, 17, 18, 19, 22, 24, 25,
                              27, 28, 29, 31, 32, 33, 34
                          ]]
        else:
            raise RuntimeError("Cannot handle MGGA deriv %d" % deriv)

        grad_inps = _get_grad_inps(grad_res, inps, derivs, ctx.needs_input_grad,
                                   deriv_idxs)
        return (*grad_inps, None, None)


class CalcMGGALibXCPol(torch.autograd.Function):
    """Calculates the energy density or its derivative w.r.t. density for
    polarized meta-GGA.

    Meta-generalized-gradient approximations (meta-GGA) are a class of approximations
    to the exchange–correlation (XC) energy functional in density functional theory
    (DFT) that depend not only upon the value of the electronic density at each
    point in space and its gradient, but also upon the Laplacian of the density
    and the kinetic energy density.

    Examples
    --------
    >>> import torch
    >>> import pylibxc
    >>> libxcfcn = pylibxc.LibXCFunctional("mgga_c_m06_l", "polarized")
    >>> rho_u = torch.tensor([0.1, 0.2, 0.3])
    >>> rho_d = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma_uu = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma_ud = torch.tensor([0.1, 0.2, 0.3])
    >>> sigma_dd = torch.tensor([0.1, 0.2, 0.3])
    >>> lapl_u = torch.tensor([0.1, 0.2, 0.3])
    >>> lapl_d = torch.tensor([0.1, 0.2, 0.3])
    >>> kin_u = torch.tensor([0.1, 0.2, 0.3])
    >>> kin_d = torch.tensor([0.1, 0.2, 0.3])
    >>> res = CalcMGGALibXCPol.apply(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd,
    ...                               lapl_u, lapl_d, kin_u, kin_d, 0, libxcfcn)[0]
    >>> print(res)
    tensor([[-0.0065, -0.0115, -0.0162]], dtype=torch.float64)

    """

    @staticmethod
    def forward(ctx, rho_u: torch.Tensor, rho_d: torch.Tensor,  # type: ignore
                sigma_uu: torch.Tensor, sigma_ud: torch.Tensor, sigma_dd: torch.Tensor,
                lapl_u: torch.Tensor, lapl_d: torch.Tensor,
                kin_u: torch.Tensor, kin_d: torch.Tensor,
                deriv: int, libxcfcn: pylibxc.functional.LibXCFunctional) -> \
            Tuple[torch.Tensor, ...]:  # type: ignore
        """Calculates and returns the energy density or its derivative w.r.t.
        density and contracted gradient and laplacian and kinetic energy density.
        Every element in the tuple is a tensor with shape of (nderiv, ninps)
        where nderiv depends on the number of derivatives for spin-up and
        spin-down combinations, e.g. nderiv == 3 for vsigma (see libxc manual)

        Parameters
        ----------
        rho_u : torch.Tensor
            The density tensor for spin-up with shape (ninps)
        rho_d : torch.Tensor
            The density tensor for spin-down with shape (ninps)
        sigma_uu : torch.Tensor
            The contracted gradient tensor for spin-up with shape (ninps)
        sigma_ud : torch.Tensor
            The contracted gradient tensor for spin-up and spin-down with shape (ninps)
        sigma_dd : torch.Tensor
            The contracted gradient tensor for spin-down with shape (ninps)
        lapl_u : torch.Tensor
            The laplacian tensor for spin-up with shape (ninps)
        lapl_d : torch.Tensor
            The laplacian tensor for spin-down with shape (ninps)
        kin_u : torch.Tensor
            The kinetic energy density tensor for spin-up with shape (ninps)
        kin_d : torch.Tensor
            The kinetic energy density tensor for spin-down with shape (ninps)
        deriv : int
            The derivative order. 0 for energy density, 1 for derivative w.r.t.
            density, 2 for second derivative w.r.t. density, etc.
        libxcfcn : pylibxc.functional.LibXCFunctional
            The libxc functional to use

        Returns
        -------
        Tuple[torch.Tensor]
            The result is a tensor with shape (nderiv, ninps)
            The result is a tensor with shape (nderiv, ninps) where the first
            dimension indicates the result for derivatives of spin-up and
            spin-down and some of its combination.

        """
        inp = {
            "rho": _pack_input(rho_u, rho_d),
            "sigma": _pack_input(sigma_uu, sigma_ud, sigma_dd),
            "lapl": _pack_input(lapl_u, lapl_d),
            "tau": _pack_input(kin_u, kin_d),
        }
        res = _get_libxc_res(inp, deriv, libxcfcn, family=4, polarized=True)

        ctx.save_for_backward(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd,
                              lapl_u, lapl_d, kin_u, kin_d, *res)
        ctx.deriv = deriv
        ctx.libxcfcn = libxcfcn
        return (*res,)

    @staticmethod
    def backward(ctx, *grad_res: torch.Tensor) -> \
            Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Returns the gradient w.r.t. the input rho and sigma.

        Parameters
        ----------
        grad_res : torch.Tensor
            The gradient of the result w.r.t. the result itself.

        Returns
        -------
        Tuple[torch.Tensor]
            The gradient w.r.t. the input rho and sigma.

        """
        inps = ctx.saved_tensors[:9]
        res = ctx.saved_tensors[9:]  # noqa: F841
        deriv = ctx.deriv
        libxcfcn = ctx.libxcfcn

        derivs = CalcMGGALibXCPol.apply(*inps, deriv + 1, libxcfcn)

        # generated by `_generate_spin_list(deriv, ["rho", "sigma"], [2, 3])`
        if deriv == 0:
            deriv_idxs = [[0], [0], [1], [1], [1], [2], [2], [3], [3]]
            spin_idxs: List[List[Tuple[int, ...]]] = [[(0,)], [(1,)], [(0,)],
                                                      [(1,)], [(2,)], [(0,)],
                                                      [(1,)], [(0,)], [(1,)]]
        elif deriv == 1:
            deriv_idxs = [[0, 1, 2, 3], [0, 1, 2, 3], [1, 4, 5,
                                                       6], [1, 4, 5, 6],
                          [1, 4, 5, 6], [2, 5, 7, 8], [2, 5, 7, 8],
                          [3, 6, 8, 9], [3, 6, 8, 9]]
            spin_idxs = [[(0, 1), (0, 1, 2), (0, 1), (0, 1)],
                         [(1, 2), (3, 4, 5), (2, 3), (2, 3)],
                         [(0, 3), (0, 1, 2), (0, 1), (0, 1)],
                         [(1, 4), (1, 3, 4), (2, 3), (2, 3)],
                         [(2, 5), (2, 4, 5), (4, 5), (4, 5)],
                         [(0, 2), (0, 2, 4), (0, 1), (0, 1)],
                         [(1, 3), (1, 3, 5), (1, 2), (2, 3)],
                         [(0, 2), (0, 2, 4), (0, 2), (0, 1)],
                         [(1, 3), (1, 3, 5), (1, 3), (1, 2)]]
        elif deriv == 2:
            deriv_idxs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          [1, 4, 5, 6, 10, 11, 12, 13, 14, 15],
                          [1, 4, 5, 6, 10, 11, 12, 13, 14, 15],
                          [1, 4, 5, 6, 10, 11, 12, 13, 14, 15],
                          [2, 5, 7, 8, 11, 13, 14, 16, 17, 18],
                          [2, 5, 7, 8, 11, 13, 14, 16, 17, 18],
                          [3, 6, 8, 9, 12, 14, 15, 17, 18, 19],
                          [3, 6, 8, 9, 12, 14, 15, 17, 18, 19]]
            spin_idxs = [[(0, 1, 2), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3),
                          (0, 1, 2, 3), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5),
                          (0, 1, 2, 3, 4, 5), (0, 1, 2), (0, 1, 2, 3),
                          (0, 1, 2)],
                         [(1, 2, 3), (3, 4, 5, 6, 7, 8), (2, 3, 4, 5),
                          (2, 3, 4, 5), (6, 7, 8, 9, 10, 11),
                          (6, 7, 8, 9, 10, 11), (6, 7, 8, 9, 10, 11), (3, 4, 5),
                          (4, 5, 6, 7), (3, 4, 5)],
                         [(0, 3, 6), (0, 1, 2, 6, 7, 8), (0, 1, 6, 7),
                          (0, 1, 6, 7), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5),
                          (0, 1, 2, 3, 4, 5), (0, 1, 2), (0, 1, 2, 3),
                          (0, 1, 2)],
                         [(1, 4, 7), (1, 3, 4, 7, 9, 10), (2, 3, 8, 9),
                          (2, 3, 8, 9), (1, 3, 4, 6, 7, 8), (2, 3, 6, 7, 8, 9),
                          (2, 3, 6, 7, 8, 9), (3, 4, 5), (4, 5, 6, 7),
                          (3, 4, 5)],
                         [(2, 5, 8), (2, 4, 5, 8, 10, 11), (4, 5, 10, 11),
                          (4, 5, 10, 11), (2, 4, 5, 7, 8, 9),
                          (4, 5, 8, 9, 10, 11), (4, 5, 8, 9, 10, 11), (6, 7, 8),
                          (8, 9, 10, 11), (6, 7, 8)],
                         [(0, 2, 4), (0, 2, 4, 6, 8, 10), (0, 1, 3, 4),
                          (0, 1, 4, 5), (0, 2, 4, 6, 8, 10), (0, 1, 3, 4, 6, 7),
                          (0, 1, 4, 5, 8, 9), (0, 1, 2), (0, 1, 2, 3),
                          (0, 1, 2)],
                         [(1, 3, 5), (1, 3, 5, 7, 9, 11), (1, 2, 4, 5),
                          (2, 3, 6, 7), (1, 3, 5, 7, 9, 11), (1, 2, 4, 5, 7, 8),
                          (2, 3, 6, 7, 10, 11), (1, 2, 3), (2, 3, 4, 5),
                          (3, 4, 5)],
                         [(0, 2, 4), (0, 2, 4, 6, 8, 10), (0, 2, 4, 6),
                          (0, 1, 3, 4), (0, 2, 4, 6, 8, 10),
                          (0, 2, 4, 6, 8, 10), (0, 1, 3, 4, 6, 7), (0, 2, 4),
                          (0, 1, 3, 4), (0, 1, 2)],
                         [(1, 3, 5), (1, 3, 5, 7, 9, 11), (1, 3, 5, 7),
                          (1, 2, 4, 5), (1, 3, 5, 7, 9, 11),
                          (1, 3, 5, 7, 9, 11), (1, 2, 4, 5, 7, 8), (1, 3, 5),
                          (1, 2, 4, 5), (1, 2, 3)]]
        else:
            raise RuntimeError(
                f"Unimplemented derivative for deriv == {deriv} for polarized MGGA"
            )

        grad_inps = _get_grad_inps(grad_res, inps, derivs, ctx.needs_input_grad,
                                   deriv_idxs, spin_idxs)
        return (*grad_inps, None, None)


def _get_libxc_res(inp: Mapping[str, Union[np.ndarray, Tuple[np.ndarray, ...],
                                           torch.Tensor, Tuple[torch.Tensor,
                                                               ...]]],
                   deriv: int, libxcfcn: pylibxc.functional.LibXCFunctional,
                   family: int, polarized: bool) -> Tuple[torch.Tensor, ...]:
    """
    Calls the libxc functional to calculate the energy density or its derivative
    w.r.t. density and contracted gradient.

    Examples
    --------
    >>> import torch
    >>> import pylibxc
    >>> libxcfcn = pylibxc.LibXCFunctional("lda_x", "unpolarized")
    >>> rho = torch.tensor([0.1, 0.2, 0.3])
    >>> res = _get_libxc_res({"rho": rho}, 0, libxcfcn, 2, False)
    >>> print(res)
    (tensor([[-0.0343, -0.0864, -0.1483]], dtype=torch.float64),)

    Parameters
    ----------
    inp: Mapping[str, Union[np.ndarray, Tuple[np.ndarray, ...], torch.Tensor, Tuple[torch.Tensor, ...]]]
        Input data for the libxc functional
    deriv: int
        Derivative order. 0 for energy density, 1 for derivative w.r.t.
        density, 2 for second derivative w.r.t. density, etc.
    libxcfcn: pylibxc.functional.LibXCFunctional
        libxc functional to use
    family: int
        Family of the functional. 2 for GGA and 4 for MGGA
    polarized: bool
        Whether the calculation is for polarized or unpolarized data

    Returns
    -------
    Tuple[torch.Tensor]
        Result is a tuple of tensors with shape (ninps)
        where the first element is the result for the energy density or its
        derivative w.r.t. density and the rest are the contracted gradient.

    """
    do_exc, do_vxc, do_fxc, do_kxc, do_lxc = _get_dos(deriv)

    res = libxcfcn.compute(inp,
                           do_exc=do_exc,
                           do_vxc=do_vxc,
                           do_fxc=do_fxc,
                           do_kxc=do_kxc,
                           do_lxc=do_lxc)

    # compile the results in a tuple with order given in the *_KEYS (e.g. LDA_KEYS)
    res = _extract_returns(res, deriv, family)

    # In libxc, "zk" is the only one returning the energy density
    # per unit volume PER UNIT PARTICLE.
    # everything else is represented by the energy density per unit volume
    # only.
    if deriv == 0:
        rho = inp["rho"]
        if polarized:
            assert isinstance(rho, np.ndarray)
            start = np.zeros(1, dtype=rho.dtype)
            rho = sum(_unpack_input(rho), start)  # rho[:, 0] + rho[:, 1]
        res0 = res[0] * rho
        res = (res0, *res[1:])

    return res


def _pack_input(*vals: torch.Tensor) -> np.ndarray:
    """Arrange the values in a numpy array with fortran memory order

    Examples
    --------
    >>> rho = torch.tensor([[1, 2], [3, 4]])
    >>> sigma = torch.tensor([[1, 2], [3, 4]])
    >>> _pack_input(rho, sigma)
    array([[[1, 1],
            [3, 3]],
    <BLANKLINE>
           [[2, 2],
            [4, 4]]])

    Parameters
    ----------
    vals: torch.Tensor
        Input values

    Returns
    -------
    np.ndarray
        Input values in a numpy array with fortran memory order

    """
    vals_np = np.asarray([val.detach().numpy() for val in vals])
    return np.ascontiguousarray(vals_np.T)


def _unpack_input(inp: np.ndarray) -> Iterator[np.ndarray]:
    """unpack from libxc input format into tuple of inputs

    Examples
    --------
    >>> inp = np.array([[1, 3], [2, 4]])
    >>> tuple(_unpack_input(inp))
    (array([1, 2]), array([3, 4]))

    Parameters
    ----------
    inp: np.ndarray
        Input values in a numpy array with fortran memory order

    Returns
    -------
    Iterator[np.ndarray]
        Unpacked input values

    """
    return (a for a in inp.T)


def _get_dos(deriv: int) -> Tuple[bool, ...]:
    """get the boolean flags for the derivatives

    Examples
    --------
    >>> _get_dos(0)
    (True, False, False, False, False)
    >>> _get_dos(1)
    (False, True, False, False, False)

    Parameters
    ----------
    deriv: int
        Derivative order. 0 for energy density, 1 for derivative w.r.t.
        density, 2 for second derivative w.r.t. density, etc.

    Returns
    -------
    Tuple[bool]
        Boolean flags for the derivatives

    """
    do_exc = deriv == 0
    do_vxc = deriv == 1
    do_fxc = deriv == 2
    do_kxc = deriv == 3
    do_lxc = deriv == 4
    return do_exc, do_vxc, do_fxc, do_kxc, do_lxc


# generated by [_generate_keys(i, ["rho", "sigma"]) for i in range(5)]
# _generate_keys function is below
LDA_KEYS = [["zk"], ["vrho"], ["v2rho2"], ["v3rho3"], ["v4rho4"]]
GGA_KEYS = [["zk"], ["vrho", "vsigma"], ["v2rho2", "v2rhosigma", "v2sigma2"],
            ["v3rho3", "v3rho2sigma", "v3rhosigma2", "v3sigma3"],
            [
                "v4rho4", "v4rho3sigma", "v4rho2sigma2", "v4rhosigma3",
                "v4sigma4"
            ]]
MGGA_KEYS = [['zk'], ['vrho', 'vsigma', 'vlapl', 'vtau'],
             [
                 'v2rho2', 'v2rhosigma', 'v2rholapl', 'v2rhotau', 'v2sigma2',
                 'v2sigmalapl', 'v2sigmatau', 'v2lapl2', 'v2lapltau', 'v2tau2'
             ],
             [
                 'v3rho3', 'v3rho2sigma', 'v3rho2lapl', 'v3rho2tau',
                 'v3rhosigma2', 'v3rhosigmalapl', 'v3rhosigmatau', 'v3rholapl2',
                 'v3rholapltau', 'v3rhotau2', 'v3sigma3', 'v3sigma2lapl',
                 'v3sigma2tau', 'v3sigmalapl2', 'v3sigmalapltau', 'v3sigmatau2',
                 'v3lapl3', 'v3lapl2tau', 'v3lapltau2', 'v3tau3'
             ],
             [
                 'v4rho4', 'v4rho3sigma', 'v4rho3lapl', 'v4rho3tau',
                 'v4rho2sigma2', 'v4rho2sigmalapl', 'v4rho2sigmatau',
                 'v4rho2lapl2', 'v4rho2lapltau', 'v4rho2tau2', 'v4rhosigma3',
                 'v4rhosigma2lapl', 'v4rhosigma2tau', 'v4rhosigmalapl2',
                 'v4rhosigmalapltau', 'v4rhosigmatau2', 'v4rholapl3',
                 'v4rholapl2tau', 'v4rholapltau2', 'v4rhotau3', 'v4sigma4',
                 'v4sigma3lapl', 'v4sigma3tau', 'v4sigma2lapl2',
                 'v4sigma2lapltau', 'v4sigma2tau2', 'v4sigmalapl3',
                 'v4sigmalapl2tau', 'v4sigmalapltau2', 'v4sigmatau3', 'v4lapl4',
                 'v4lapl3tau', 'v4lapl2tau2', 'v4lapltau3', 'v4tau4'
             ]]


def _extract_returns(ret: Mapping[str, np.ndarray], deriv: int, family: int) -> \
        Tuple[torch.Tensor, ...]:
    """Compile the returns from pylibxc into a tuple of tensors with order given
    by the keys

    Examples
    --------
    >>> import numpy as np
    >>> ret = {"zk": np.array([1, 2, 3])}
    >>> _extract_returns(ret, 0, 1)
    (tensor([1, 2, 3]),)

    Parameters
    ----------
    ret: Mapping[str, np.ndarray]
        Return values from pylibxc
    deriv: int
        Derivative order. 0 for energy density, 1 for derivative w.r.t.
        density, 2 for second derivative w.r.t. density, etc.
    family: int
        Family of the functional. 1 for LDA, 2 for GGA, 4 for MGGA

    Returns
    -------
    Tuple[torch.Tensor]
        Result is a tuple of tensors with shape (ninps)
        where the first element is the result for the energy density or its
        derivative w.r.t. density and the rest are the contracted gradient.

    """

    def a(v):
        return torch.as_tensor(v.T)

    if family == 1:
        keys = LDA_KEYS
    elif family == 2:
        keys = GGA_KEYS
    elif family == 4:
        keys = MGGA_KEYS
    else:
        raise RuntimeError("Unknown libxc family %d" % family)
    return tuple(a(ret[key]) for key in keys[deriv])


def _get_grad_inps(
    grad_res: Tuple[torch.Tensor, ...],
    inps: Tuple[torch.Tensor, ...],
    derivs: Tuple[torch.Tensor, ...],
    needs_input_grad: List[bool],
    deriv_idxs: List[List[int]],
    spin_idxs: Optional[List[List[Tuple[int, ...]]]] = None
) -> Tuple[Optional[torch.Tensor], ...]:
    """Calculate the grad_inp from grad_res and given deriv_idxs
    each row indicates the input, while the column indicates the index in out
    deriv_idxs[i][j] means that grad_inp[i] += grad_res[j] * derivs[deriv_idxs[i][j]]

    Examples
    --------
    >>> grad_res = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    >>> inps = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    >>> derivs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    >>> needs_input_grad = [True, True]
    >>> deriv_idxs = [[0], [1]]
    >>> _get_grad_inps(grad_res, inps, derivs, needs_input_grad, deriv_idxs)
    (tensor([1, 4, 9]), tensor([ 4, 10, 18]))

    Parameters
    ----------
    grad_res: Tuple[torch.Tensor]
        Gradient of the result w.r.t. the result itself.
    inps: Tuple[torch.Tensor]
        Input tensors
    derivs: Tuple[torch.Tensor]
        Derivative tensors
    needs_input_grad: List[bool]
        Boolean list indicating whether the input requires grad
    deriv_idxs: List[List[int]]
        List of indices for the derivatives
    spin_idxs: Optional[List[List[Tuple[int, ...]]], optional
        List of spin indices, by default None

    Returns
    -------
    Tuple[Optional[torch.Tensor]]
        Gradient w.r.t. the inputs

    """
    grad_inps: List[Optional[torch.Tensor]] = []
    for i in range(len(deriv_idxs)):
        # if the input does not requires grad, then don't compute
        if not needs_input_grad[i]:
            grad_inps.append(None)
            continue

        grad_inp = torch.zeros_like(inps[i])
        didxs = deriv_idxs[i]
        if spin_idxs is not None:
            sidxs = spin_idxs[i]
        for j in range(len(didxs)):
            if spin_idxs is None:
                grad_inp = grad_inp + grad_res[j] * derivs[didxs[j]]
            else:
                grad_inp = grad_inp + torch.sum(
                    grad_res[j] * derivs[didxs[j]][sidxs[j], :], dim=0)
        grad_inps.append(grad_inp)
    return tuple(grad_inps)
