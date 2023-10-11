"""
Density Functional Theory BaseXC Classes.
Derived from: https://github.com/diffqc/dqc/blob/master/dqc/xc/base_xc.py
"""

from contextlib import contextmanager
from abc import abstractmethod
import torch
import xitorch as xt
from typing import List, Union, overload, Iterator
from deepchem.utils.dft_utils.datastruct import ValGrad, SpinParam


class BaseXC(xt.EditableModule):
    """XC is class that calculates the components of xc potential and energy
    density given the density.

    Exchange-correlation (xc) potential is a term used in density functional
    theory to describe the relationship between the electron density and the
    exchange and correlation energy.

    Example
    -------
    >>> import torch
    >>> from deepchem.utils.dft_utils.xc.base_xc import BaseXC
    >>> from deepchem.utils.dft_utils.datastruct import ValGrad
    >>> class MyXC(BaseXC):
    ...     def get_edensityxc(self, densinfo: ValGrad) -> torch.Tensor:
    ...         return densinfo.value ** 2
    >>> xc = MyXC()
    >>> densinfo = ValGrad(value=torch.tensor([1., 2., 3.]), grad=torch.tensor([4., 5., 6.]))
    >>> xc.get_edensityxc(densinfo)
    tensor([1., 4., 9.])


    """

    @abstractmethod
    def family(self) -> int:
        """Returns 1 for LDA, 2 for GGA, and 4 for Meta-GGA."""
        pass

    @abstractmethod
    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the xc energy density (energy per unit volume).

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            The density information. If the xc is unpolarized, then densinfo
            is ValGrad. If the xc is polarized, then densinfo is SpinParam[ValGrad].

        Returns
        -------
        torch.Tensor
            The energy density.

        """

        pass

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad:
        ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]:
        ...

    def get_vxc(self, densinfo):
        """Returns the ValGrad for the xc potential given the density info
        for unpolarized case.

        This is the default implementation of vxc if there is no implementation
        in the specific class of XC.

        parameters
        ----------
        densinfo
            The density information. If the xc is unpolarized, then densinfo
            is ValGrad. If the xc is polarized, then densinfo is SpinParam[ValGrad].

        returns
        -------
        ValGrad or SpinParam[ValGrad]
            The ValGrad for the xc potential.

        """

        # mark the densinfo components as requiring grads
        with self._enable_grad_densinfo(densinfo):
            with torch.enable_grad():
                edensity = self.get_edensityxc(densinfo)  # (*BD, nr)
            grad_outputs = torch.ones_like(edensity)
            grad_enabled = torch.is_grad_enabled()

            if not isinstance(densinfo, ValGrad):  # polarized case
                if self.family() == 1:  # LDA
                    params = (densinfo.u.value, densinfo.d.value)
                    dedn_u, dedn_d = torch.autograd.grad(
                        edensity,
                        params,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs)

                    return SpinParam(u=ValGrad(value=dedn_u),
                                     d=ValGrad(value=dedn_d))

                elif self.family() == 2:  # GGA
                    params = (densinfo.u.value, densinfo.d.value,
                              densinfo.u.grad, densinfo.d.grad)
                    dedn_u, dedn_d, dedg_u, dedg_d = torch.autograd.grad(
                        edensity,
                        params,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs)

                    return SpinParam(u=ValGrad(value=dedn_u, grad=dedg_u),
                                     d=ValGrad(value=dedn_d, grad=dedg_d))

                elif self.family() == 4:
                    params = (densinfo.u.value, densinfo.d.value,
                              densinfo.u.grad, densinfo.d.grad, densinfo.u.lapl,
                              densinfo.d.lapl, densinfo.u.kin, densinfo.d.kin)
                    dedn_u, dedn_d, dedg_u, dedg_d, dedl_u, dedl_d, dedk_u, dedk_d = torch.autograd.grad(
                        edensity,
                        params,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs,
                        allow_unused=True)

                    # mgga might only use one of either lapl or kin, so we need to change the deriv manually to 0s
                    dedl_u = dedl_u if dedl_u is not None else torch.zeros_like(
                        dedn_u)
                    dedk_u = dedk_u if dedk_u is not None else torch.zeros_like(
                        dedn_u)
                    dedl_d = dedl_d if dedl_d is not None else torch.zeros_like(
                        dedn_d)
                    dedk_d = dedk_d if dedk_d is not None else torch.zeros_like(
                        dedn_d)

                    return SpinParam(u=ValGrad(value=dedn_u,
                                               grad=dedg_u,
                                               lapl=dedl_u,
                                               kin=dedk_u),
                                     d=ValGrad(value=dedn_d,
                                               grad=dedg_d,
                                               lapl=dedl_d,
                                               kin=dedk_d))

                else:
                    raise NotImplementedError(
                        "Default polarized vxc for family %s is not implemented"
                        % self.family())

            else:  # unpolarized case
                if self.family() == 1:  # LDA
                    dedn, = torch.autograd.grad(edensity,
                                                densinfo.value,
                                                create_graph=grad_enabled,
                                                grad_outputs=grad_outputs)

                    return ValGrad(value=dedn)

                elif self.family() == 2:  # GGA
                    dedn, dedg = torch.autograd.grad(
                        edensity, (densinfo.value, densinfo.grad),
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs)

                    return ValGrad(value=dedn, grad=dedg)

                elif self.family() == 4:  # MGGA
                    dedn, dedg, dedl, dedk = torch.autograd.grad(
                        edensity, (densinfo.value, densinfo.grad, densinfo.lapl,
                                   densinfo.kin),
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs,
                        allow_unused=True)

                    # mgga might only use one of either lapl or kin, so we need to change the deriv manually to 0s
                    dedl = dedl if dedl is not None else torch.zeros_like(dedn)
                    dedk = dedk if dedk is not None else torch.zeros_like(dedn)

                    return ValGrad(value=dedn, grad=dedg, lapl=dedl, kin=dedk)

                else:
                    raise NotImplementedError(
                        "Default vxc for family %d is not implemented" %
                        self.family())

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_vxc":
            return self.getparamnames("get_edensityxc", prefix=prefix)
        else:
            raise KeyError("Unknown methodname: %s" % methodname)

    @contextmanager
    def _enable_grad_densinfo(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> Iterator:
        # set the context where some elements (depends on xc family) in densinfo requires grad

        def _get_set_grad(vars: List[torch.Tensor]) -> List[bool]:
            # set the vars to require grad and returns the previous state of the vars
            reqgrads = []
            for var in vars:
                reqgrads.append(var.requires_grad)
                var.requires_grad_()
            return reqgrads

        def _restore_grad(reqgrads: List[bool],
                          vars: List[torch.Tensor]) -> None:
            # restore the state of requiring grad based on reqgrads list
            # all vars before this function requires grad
            for reqgrad, var in zip(reqgrads, vars):
                if not reqgrad:
                    var.requires_grad_(False)

        # getting which parameters should require grad
        if not isinstance(densinfo, ValGrad):  # a spinparam
            params = [densinfo.u.value, densinfo.d.value]
            if self.family() >= 2:  # GGA
                assert densinfo.u.grad is not None
                assert densinfo.d.grad is not None
                params.extend([densinfo.u.grad, densinfo.d.grad])
            if self.family() >= 3:  # MGGA
                assert densinfo.u.lapl is not None
                assert densinfo.d.lapl is not None
                assert densinfo.u.kin is not None
                assert densinfo.d.kin is not None
                params.extend([
                    densinfo.u.lapl, densinfo.d.lapl, densinfo.u.kin,
                    densinfo.d.kin
                ])
        else:
            params = [densinfo.value]
            if self.family() >= 2:
                assert densinfo.grad is not None
                params.append(densinfo.grad)
            if self.family() >= 3:
                assert densinfo.lapl is not None
                assert densinfo.kin is not None
                params.extend([densinfo.lapl, densinfo.kin])

        try:
            # set the params to require grad
            reqgrads = _get_set_grad(params)
            yield
        finally:
            _restore_grad(reqgrads, params)

    # special operations
    def __add__(self, other):
        return AddBaseXC(self, other)

    def __mul__(self, other: Union[float, int, torch.Tensor]):
        if isinstance(other, float) or isinstance(other, int):
            return MulBaseXC(self, float(other))
        elif isinstance(other, torch.Tensor):
            return MulBaseXC(self, other)
        else:
            raise ValueError(
                "BaseXC can only be multiplied with float or tensor")

    def __rmul__(self, other: Union[float, int, torch.Tensor]):
        return self.__mul__(other)


class AddBaseXC(BaseXC):
    """AddBaseXC is the class for adding two BaseXC together. This is useful
    for combining two xc potentials together.

    Example
    -------
    >>> import torch
    >>> from deepchem.utils.dft_utils.xc import BaseXC
    >>> from deepchem.utils.dft_utils.datastruct import ValGrad
    >>> class MyXC(BaseXC):
    ...     def family(self):
    ...         return 1
    ...     def get_edensityxc(self, densinfo: ValGrad) -> torch.Tensor:
    ...         return densinfo.value ** 2
    >>> xc = MyXC()
    >>> densinfo = ValGrad(value=torch.tensor([1., 2., 3.]), grad=torch.tensor([4., 5., 6.]))
    >>> xc.get_edensityxc(densinfo)
    tensor([1., 4., 9.])
    >>> xc2 = MyXC()
    >>> xc2.get_edensityxc(densinfo)
    tensor([1., 4., 9.])
    >>> xc3 = xc + xc2
    >>> xc3.get_edensityxc(densinfo)
    tensor([ 2.,  8., 18.])

    """

    def __init__(self, a: BaseXC, b: BaseXC) -> None:
        self.a = a
        self.b = b
        self._family = max(a.family(), b.family())

    @property
    def family(self):
        return self._family

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad:
        ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]:
        ...

    def get_vxc(self, densinfo):
        avxc = self.a.get_vxc(densinfo)
        bvxc = self.b.get_vxc(densinfo)

        if isinstance(densinfo, ValGrad):
            return avxc + bvxc
        else:
            return SpinParam(u=avxc.u + bvxc.u, d=avxc.d + bvxc.d)

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> \
            torch.Tensor:
        return self.a.get_edensityxc(densinfo) + self.b.get_edensityxc(densinfo)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return self.a.getparamnames(methodname, prefix=prefix + "a.") + \
            self.b.getparamnames(methodname, prefix=prefix + "b.")


class MulBaseXC(BaseXC):
    """MulBaseXC is the class for multiplying BaseXC with a float or tensor.
    This is useful for scaling the xc potential.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.xc import BaseXC
    >>> from deepchem.utils.dft_utils.datastruct import ValGrad
    >>> class MyXC(BaseXC):
    ...     def family(self):
    ...         return 1
    ...     def get_edensityxc(self, densinfo: ValGrad) -> torch.Tensor:
    ...         return densinfo.value ** 2
    >>> xc = MyXC()
    >>> densinfo = ValGrad(value=torch.tensor([1., 2., 3.]), grad=torch.tensor([4., 5., 6.]))
    >>> xc.get_edensityxc(densinfo)
    tensor([1., 4., 9.])
    >>> xc2 = xc * 2.0
    >>> xc2.get_edensityxc(densinfo)
    tensor([ 2.,  8., 18.])

    """

    def __init__(self, a: BaseXC, b: Union[float, torch.Tensor]) -> None:
        self.a = a
        self.b = b
        if isinstance(b, torch.Tensor):
            msg = "XC multiplication with tensor can only be done with 1-element tensor"
            assert b.numel() == 1, msg

    @property
    def family(self):
        return self.a.family

    @overload
    def get_vxc(self, densinfo: ValGrad) -> ValGrad:
        ...

    @overload
    def get_vxc(self, densinfo: SpinParam[ValGrad]) -> SpinParam[ValGrad]:
        ...

    def get_vxc(self, densinfo):
        avxc = self.a.get_vxc(densinfo)

        if isinstance(densinfo, ValGrad):
            return avxc * self.b
        else:
            return SpinParam(u=avxc.u * self.b, d=avxc.d * self.b)

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> \
            torch.Tensor:
        return self.a.get_edensityxc(densinfo) * self.b

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        params = self.a.getparamnames(methodname, prefix=prefix + "a.")
        if isinstance(self.b, torch.Tensor):
            params = params + [prefix + "b"]
        return params
