from contextlib import contextmanager
from abc import abstractmethod, abstractproperty
import torch
from deepchem.utils.differentiation_utils import EditableModule
from typing import List, Union, Iterator
from deepchem.utils.dft_utils import ValGrad, SpinParam


class BaseXC(EditableModule):
    """
    This is the base class for the exchange-correlation (XC) functional.
    The XC functional is used to calculate the exchange-correlation energy
    and potential. The XC functional is usually divided into three families:
    LDA, GGA, and Meta-GGA. The LDA is the simplest one, which only depends
    on the density. The GGA depends on the density and its gradient. The
    Meta-GGA depends on the density, its gradient, and its Laplacian.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import ValGrad, SpinParam
    >>> from deepchem.utils.dft_utils import BaseXC
    >>> class MyXC(BaseXC):
    ...     @property
    ...     def family(self) -> int:
    ...         return 1
    ...     def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
    ...         if isinstance(densinfo, ValGrad):
    ...             return densinfo.value.pow(2)
    ...         else:
    ...             return densinfo.u.value.pow(2) + densinfo.d.value.pow(2)
    ...     def get_vxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> Union[ValGrad, SpinParam[ValGrad]]:
    ...         if isinstance(densinfo, ValGrad):
    ...             return ValGrad(value=2*densinfo.value)
    ...         else:
    ...             return SpinParam(u=ValGrad(value=2*densinfo.u.value),
    ...                              d=ValGrad(value=2*densinfo.d.value))
    >>> xc = MyXC()
    >>> densinfo = ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True))
    >>> xc.get_edensityxc(densinfo)
    tensor([1., 4., 9.], grad_fn=<PowBackward0>)
    >>> xc.get_vxc(densinfo)
    ValGrad(value=tensor([2., 4., 6.], grad_fn=<MulBackward0>), grad=None, lapl=None, kin=None)
    >>> densinfo = SpinParam(u=ValGrad(value=torch.tensor([1., 2., 3.], requires_grad=True)),
    ...                      d=ValGrad(value=torch.tensor([4., 5., 6.], requires_grad=True)))
    >>> xc.get_edensityxc(densinfo)
    tensor([17., 29., 45.], grad_fn=<AddBackward0>)
    >>> xc.get_vxc(densinfo)
    SpinParam(u=ValGrad(value=tensor([2., 4., 6.], grad_fn=<MulBackward0>), grad=None, lapl=None, kin=None), d=ValGrad(value=tensor([ 8., 10., 12.], grad_fn=<MulBackward0>), grad=None, lapl=None, kin=None))

    """

    @abstractproperty
    def family(self) -> int:
        """Returns 1 for LDA, 2 for GGA, and 4 for Meta-GGA."""
        pass

    @abstractmethod
    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the xc energy density (energy per unit volume)

        Parameters
        ----------
        densinfo : Union[ValGrad, SpinParam[ValGrad]]
            The density information.
            If the XC is unpolarized, then densinfo is ValGrad.
            If the XC is polarized, then densinfo is SpinParam[ValGrad].
            The ValGrad contains the value and gradient of the density.
            The SpinParam[ValGrad] contains the value and gradient of the density
            for each spin channel.

        Returns
        -------
        torch.Tensor
            The energy density of the XC.

        """
        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, ndim, nr)
        # return: (*BD, nr)
        pass

    def get_vxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]):
        """
        Returns the ValGrad for the xc potential given the density info
        for unpolarized case.

        This is the default implementation of vxc if there is no implementation
        in the specific class of XC.

        Parameters
        ----------
        densinfo : Union[ValGrad, SpinParam[ValGrad]]
            The density information.
            If the XC is unpolarized, then densinfo is ValGrad.
            If the XC is polarized, then densinfo is SpinParam[ValGrad].
            The ValGrad contains the value and gradient of the density.
            The SpinParam[ValGrad] contains the value and gradient of the density
            for each spin channel.

        Returns
        -------
        Union[ValGrad, SpinParam[ValGrad]]
            The ValGrad for the xc potential.
            If the XC is unpolarized, then the return is ValGrad.
            If the XC is polarized, then the return is SpinParam[ValGrad].

        """
        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, ndim, nr)
        # return:
        # potentialinfo.value & lapl: (*BD, nr)
        # potentialinfo.grad: (*BD, ndim, nr)

        # mark the densinfo components as requiring grads
        with self._enable_grad_densinfo(densinfo):
            with torch.enable_grad():
                edensity = self.get_edensityxc(densinfo)  # (*BD, nr)
            grad_outputs = torch.ones_like(edensity)
            grad_enabled = torch.is_grad_enabled()

            if not isinstance(densinfo, ValGrad):  # polarized case
                if self.family == 1:  # LDA
                    params = (densinfo.u.value, densinfo.d.value)
                    dedn_u, dedn_d = torch.autograd.grad(
                        edensity,
                        params,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs)

                    return SpinParam(u=ValGrad(value=dedn_u),
                                     d=ValGrad(value=dedn_d))

                elif self.family == 2:  # GGA
                    params = (
                        densinfo.u.value,
                        densinfo.d.value,  # type: ignore[assignment]
                        densinfo.u.grad,
                        densinfo.d.grad)
                    dedn_u, dedn_d, dedg_u, dedg_d = torch.autograd.grad(
                        edensity,
                        params,
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs)

                    return SpinParam(u=ValGrad(value=dedn_u, grad=dedg_u),
                                     d=ValGrad(value=dedn_d, grad=dedg_d))

                elif self.family == 4:
                    params = (
                        densinfo.u.value,
                        densinfo.d.value,  # type: ignore[assignment]
                        densinfo.u.grad,
                        densinfo.d.grad,
                        densinfo.u.lapl,
                        densinfo.d.lapl,
                        densinfo.u.kin,
                        densinfo.d.kin)
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
                        % self.family)

            else:  # unpolarized case
                if self.family == 1:  # LDA
                    dedn, = torch.autograd.grad(edensity,
                                                densinfo.value,
                                                create_graph=grad_enabled,
                                                grad_outputs=grad_outputs)

                    return ValGrad(value=dedn)

                elif self.family == 2:  # GGA
                    assert densinfo.grad is not None
                    dedn, dedg = torch.autograd.grad(
                        edensity, [densinfo.value, densinfo.grad],
                        create_graph=grad_enabled,
                        grad_outputs=grad_outputs)

                    return ValGrad(value=dedn, grad=dedg)

                elif self.family == 4:  # MGGA
                    assert densinfo.grad is not None
                    assert densinfo.lapl is not None
                    assert densinfo.kin is not None
                    dedn, dedg, dedl, dedk = torch.autograd.grad(
                        edensity, [
                            densinfo.value, densinfo.grad, densinfo.lapl,
                            densinfo.kin
                        ],
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
                        self.family)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        This method should list tensor names that affect the output of the
        method with name indicated in ``methodname``.
        If the ``methodname`` is not on the list in this function, it should
        raise ``KeyError``.

        Parameters
        ---------
        methodname: str
            The name of the method of the class.
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            Sequence of name of parameters affecting the output of the method.

        Raises
        ------
        KeyError
            If the list in this function does not contain ``methodname``.

        """
        if methodname == "get_vxc":
            return self.getparamnames("get_edensityxc", prefix=prefix)
        else:
            raise KeyError("Unknown methodname: %s" % methodname)

    @contextmanager
    def _enable_grad_densinfo(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> Iterator:
        """
        set the context where some elements (depends on xc family) in densinfo requires grad

        Parameters
        ----------
        densinfo : Union[ValGrad, SpinParam[ValGrad]]
            The density information.
            If the XC is unpolarized, then densinfo is ValGrad.
            If the XC is polarized, then densinfo is SpinParam[ValGrad].

        Returns
        -------
        Iterator
            The context where some elements in densinfo requires grad

        """

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
            if self.family >= 2:  # GGA
                assert densinfo.u.grad is not None
                assert densinfo.d.grad is not None
                params.extend([densinfo.u.grad, densinfo.d.grad])
            if self.family >= 3:  # MGGA
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
            if self.family >= 2:
                assert densinfo.grad is not None
                params.append(densinfo.grad)
            if self.family >= 3:
                assert densinfo.lapl is not None
                assert densinfo.kin is not None
                params.extend([densinfo.lapl, densinfo.kin])

        try:
            # set the params to require grad
            reqgrads = _get_set_grad(params)
            yield
        finally:
            _restore_grad(reqgrads, params)
