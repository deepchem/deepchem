from typing import Callable, Mapping, Any, Sequence, Union
import torch
from deepchem.utils.differentiation_utils.optimize.rootsolver import broyden1, broyden2, \
    linearmixing
from deepchem.utils.differentiation_utils.optimize.equilibrium import anderson_acc
from deepchem.utils.differentiation_utils.optimize.minimizer import gd, adam
from deepchem.utils.differentiation_utils.solve import solve
from deepchem.utils.differentiation_utils.grad import jac
from deepchem.utils.differentiation_utils import get_pure_function, make_sibling
from deepchem.utils.differentiation_utils.misc import get_method
from deepchem.utils import TensorNonTensorSeparator

__all__ = ["equilibrium", "rootfinder", "minimize"]

_RF_METHODS = {
    "broyden1": broyden1,
    "broyden2": broyden2,
    "linearmixing": linearmixing,
}

_EQUIL_METHODS = {
    # equilibrium can use all rootfinder methods, but there are several methods developed specifically for
    # equilibrium (or fixed-point iterations). This dictionary gives the list of those special methods.
    "anderson_acc": anderson_acc,
}

_OPT_METHODS = {
    "gd": gd,
    "adam": adam,
}


def rootfinder(fcn: Callable[..., torch.Tensor],
               y0: torch.Tensor,
               params: Sequence[Any] = [],
               bck_options: Mapping[str, Any] = {},
               method: Union[str, Callable, None] = None,
               **fwd_options) -> torch.Tensor:
    r"""
    Solving the rootfinder equation of a given function,

    .. math::

        \mathbf{0} = \mathbf{f}(\mathbf{y}, \theta)

    where :math:`\mathbf{f}` is a function that can be non-linear and
    produce output of the same shape of :math:`\mathbf{y}`, and :math:`\theta`
    is other parameters required in the function.
    The output of this block is :math:`\mathbf{y}`
    that produces the :math:`\mathbf{0}` as the output.

    Parameters
    ----------
    fcn : callable
        The function :math:`\mathbf{f}` with output tensor ``(*ny)``
    y0 : torch.tensor
        Initial guess of the solution with shape ``(*ny)``
    params : list
        Sequence of any other parameters to be put in ``fcn``
    bck_options : dict
        Method-specific options for the backward solve (see :func:`xitorch.linalg.solve`)
    method : str or callable or None
        Rootfinder method. If None, it will choose ``"broyden1"``.
    **fwd_options
        Method-specific options (see method section)

    Returns
    -------
    torch.tensor
        The solution which satisfies
        :math:`\mathbf{0} = \mathbf{f}(\mathbf{y},\theta)`
        with shape ``(*ny)``

    Example
    -------
    >>> import torch
    >>> def func1(y, A):  # example function
    ...     return torch.tanh(A @ y + 0.1) + y / 2.0
    >>> A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    >>> y0 = torch.zeros((2,1))  # zeros as the initial guess
    >>> yroot = rootfinder(func1, y0, params=(A,))
    >>> print(yroot)
    tensor([[-0.0459],
            [-0.0663]], grad_fn=<_RootFinderBackward>)

    """

    pfunc = get_pure_function(fcn)
    fwd_options["method"] = _get_rootfinder_default_method(method)
    return _RootFinder.apply(pfunc, y0, pfunc, "rootfinder", fwd_options,
                             bck_options, len(params), *params,
                             *pfunc.objparams())


def equilibrium(fcn: Callable[..., torch.Tensor],
                y0: torch.Tensor,
                params: Sequence[Any] = [],
                bck_options: Mapping[str, Any] = {},
                method: Union[str, Callable, None] = None,
                **fwd_options) -> torch.Tensor:
    r"""
    Solving the equilibrium equation of a given function,

    .. math::

        \mathbf{y} = \mathbf{f}(\mathbf{y}, \theta)

    where :math:`\mathbf{f}` is a function that can be non-linear and
    produce output of the same shape of :math:`\mathbf{y}`, and :math:`\theta`
    is other parameters required in the function.
    The output of this block is :math:`\mathbf{y}`
    that produces the same :math:`\mathbf{y}` as the output.

    Parameters
    ----------
    fcn : callable
        The function :math:`\mathbf{f}` with output tensor ``(*ny)``
    y0 : torch.tensor
        Initial guess of the solution with shape ``(*ny)``
    params : list
        Sequence of any other parameters to be put in ``fcn``
    bck_options : dict
        Method-specific options for the backward solve (see :func:`xitorch.linalg.solve`)
    method : str or None
        Rootfinder method. If None, it will choose ``"broyden1"``.
    **fwd_options
        Method-specific options (see method section)

    Returns
    -------
    torch.tensor
        The solution which satisfies
        :math:`\mathbf{y} = \mathbf{f}(\mathbf{y},\theta)`
        with shape ``(*ny)``

    Example
    -------
    >>> import torch
    >>> def func1(y, A):  # example function
    ...     return torch.tanh(A @ y + 0.1) + y / 2.0
    >>> A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    >>> y0 = torch.zeros((2,1))  # zeros as the initial guess
    >>> yequil = equilibrium(func1, y0, params=(A,))
    >>> print(yequil)
    tensor([[ 0.2313],
            [-0.5957]], grad_fn=<_RootFinderBackward>)

    Note
    ----
    * This is a direct implementation of finding the root of
      :math:`\mathbf{g}(\mathbf{y}, \theta) = \mathbf{y} - \mathbf{f}(\mathbf{y}, \theta)`

    """
    pfunc = get_pure_function(fcn)

    @make_sibling(pfunc)
    def new_fcn(y, *params):
        return y - pfunc(y, *params)

    method = _get_equilibrium_default_method(method)
    fwd_options["method"] = method
    fwd_fcn = pfunc if method in _EQUIL_METHODS else new_fcn
    alg_type = "equilibrium" if method in _EQUIL_METHODS else "rootfinder"
    return _RootFinder.apply(new_fcn, y0, fwd_fcn, alg_type, fwd_options,
                             bck_options, len(params), *params,
                             *pfunc.objparams())


def minimize(fcn: Callable[..., torch.Tensor],
             y0: torch.Tensor,
             params: Sequence[Any] = [],
             bck_options: Mapping[str, Any] = {},
             method: Union[None, str, Callable] = None,
             **fwd_options) -> torch.Tensor:
    r"""
    Solve the unbounded minimization problem:

    .. math::

        \mathbf{y^*} = \arg\min_\mathbf{y} f(\mathbf{y}, \theta)

    to find the best :math:`\mathbf{y}` that minimizes the output of the
    function :math:`f`.

    Parameters
    ----------
    fcn: callable
        The function to be optimized with output tensor with 1 element.
    y0: torch.tensor
        Initial guess of the solution with shape ``(*ny)``
    params: list
        Sequence of any other parameters to be put in ``fcn``
    bck_options: dict
        Method-specific options for the backward solve (see :func:`xitorch.linalg.solve`)
    method: str or callable or None
        Minimization method. If None, it will choose ``"broyden1"``.
    **fwd_options
        Method-specific options (see method section)

    Returns
    -------
    torch.tensor
        The solution of the minimization with shape ``(*ny)``

    Example
    -------
    >>> import torch
    >>> def func1(y, A):  # example function
    ...     return torch.sum((A @ y)**2 + y / 2.0)
    >>> A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    >>> y0 = torch.zeros((2,1))  # zeros as the initial guess
    >>> ymin = minimize(func1, y0, params=(A,))
    >>> print(ymin)
    tensor([[-0.0519],
            [-0.2684]], grad_fn=<_RootFinderBackward>)

    """

    assert not torch.is_complex(y0), \
        "complex number is not supported on xitorch.optimize.rootfinder at the moment"

    pfunc = get_pure_function(fcn)

    fwd_options["method"] = _get_minimizer_default_method(method)
    method = fwd_options["method"]

    # minimization can use rootfinder algorithm, so check if it is actually
    # using the optimization algorithm, not the rootfinder algorithm
    opt_method = method not in _RF_METHODS.keys()

    # the rootfinder algorithms are designed to move to the opposite direction
    # of the output of the function, so the output of this function is just
    # the grad of z w.r.t. y
    # if it is going to optimization method, then also returns the value

    @make_sibling(pfunc)
    def _min_fwd_fcn(y, *params):
        with torch.enable_grad():
            y1 = y.clone().requires_grad_()
            z = pfunc(y1, *params)
        grady, = torch.autograd.grad(z, (y1,),
                                     retain_graph=True,
                                     create_graph=torch.is_grad_enabled())
        return z, grady

    @make_sibling(_min_fwd_fcn)
    def _rf_fcn(y, *params):
        z, grady = _min_fwd_fcn(y, *params)
        return grady

    # if using the optimization algorithm, then the forward function is the one
    # that returns f and grad
    if opt_method:
        _fwd_fcn = _min_fwd_fcn
    # if it is just using the rootfinder algorithm, then the forward function
    # is the one that returns only the grad
    else:
        _fwd_fcn = _rf_fcn

    alg_type = "minimizer" if opt_method else "rootfinder"
    return _RootFinder.apply(_rf_fcn, y0, _fwd_fcn, alg_type, fwd_options,
                             bck_options, len(params), *params,
                             *pfunc.objparams())


class _RootFinder(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fcn, y0, fwd_fcn, alg_type, options, bck_options, nparams,
                *allparams):
        """Forward method for the rootfinder, minimizer, and equilibrium

        Parameters
        ----------
        fcn:
            a function that returns what has to be 0 (will be used in the
            backward, not used in the forward). For minimization, it is
            the gradient
        y0:
            initial guess
        fwd_fcn:
            a function that will be executed in the forward method
            (unused in the backward)
        alg_type:
            the type of algorithm: "rootfinder", "minimizer", or "equilibrium"
        options:
            options for the forward method
        bck_options:
            options for the backward method
        nparams:
            number of parameters
        allparams:
            all parameters (including the non-tensor parameters)

        This class is also used for minimization, where fcn and fwd_fcn might
        be slightly different

        Returns
        -------
        torch.tensor
            The solution of the rootfinder, minimizer, or equilibrium

        """

        # set default options
        config = options
        ctx.bck_options = bck_options

        params = allparams[:nparams]
        objparams = allparams[nparams:]

        with fwd_fcn.useobjparams(objparams):

            method = config.pop("method")
            methods = {
                "minimizer": _OPT_METHODS,
                "rootfinder": _RF_METHODS,
                "equilibrium": _EQUIL_METHODS
            }[alg_type]
            name = alg_type
            method_fcn = get_method(name, methods, method)
            y = method_fcn(fwd_fcn, y0, params, **config)

        ctx.fcn = fcn

        # split tensors and non-tensors params
        ctx.nparams = nparams
        ctx.param_sep = TensorNonTensorSeparator(allparams)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(y, *tensor_params)

        return y

    @staticmethod
    def backward(ctx, grad_yout):
        """Backward method for the rootfinder, minimizer, and equilibrium

        Parameters
        ----------
        grad_yout: torch.tensor
            the gradient of the output of the rootfinder, minimizer, or equilibrium

        Returns
        -------
        tuple
            The gradients of the parameters

        """

        param_sep = ctx.param_sep
        yout = ctx.saved_tensors[0]
        nparams = ctx.nparams
        fcn = ctx.fcn

        # merge the tensor and nontensor parameters
        tensor_params = ctx.saved_tensors[1:]
        allparams = param_sep.reconstruct_params(tensor_params)
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        # dL/df
        with ctx.fcn.useobjparams(objparams):

            jac_dfdy = jac(fcn, params=(yout, *params), idxs=[0])[0]
            gyfcn = solve(A=jac_dfdy.H,
                          B=-grad_yout.reshape(-1, 1),
                          bck_options=ctx.bck_options,
                          **ctx.bck_options)
            gyfcn = gyfcn.reshape(grad_yout.shape)

            # get the grad for the params
            with torch.enable_grad():
                tensor_params_copy = [
                    p.clone().requires_grad_() for p in tensor_params
                ]
                allparams_copy = param_sep.reconstruct_params(
                    tensor_params_copy)
                params_copy = allparams_copy[:nparams]
                objparams_copy = allparams_copy[nparams:]
                with ctx.fcn.useobjparams(objparams_copy):
                    yfcn = fcn(yout, *params_copy)

            grad_tensor_params = torch.autograd.grad(
                yfcn,
                tensor_params_copy,
                grad_outputs=gyfcn,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True)
            grad_nontensor_params = [
                None for _ in range(param_sep.nnontensors())
            ]
            grad_params = param_sep.reconstruct_params(grad_tensor_params,
                                                       grad_nontensor_params)

        return (None, None, None, None, None, None, None, *grad_params)


def _get_rootfinder_default_method(
        method: Union[str, Callable,
                      None] = None) -> Union[str, Callable, None]:
    """Get the default method for the rootfinder, minimizer, and equilibrium

    Examples
    --------
    >>> _get_rootfinder_default_method(None)
    'broyden1'

    Parameters
    ----------
    method: str or None
        The method name

    Returns
    -------
    str
        The method name

    """
    if method is None:
        return "broyden1"
    else:
        return method


def _get_equilibrium_default_method(
        method: Union[str, Callable,
                      None] = None) -> Union[str, Callable, None]:
    """Get the default method for the equilibrium

    Examples
    --------
    >>> _get_equilibrium_default_method(None)
    'broyden1'

    Parameters
    ----------
    method: str or None
        The method name

    Returns
    -------
    str
        The method name

    """
    if method is None:
        return _get_rootfinder_default_method(method)
    else:
        return method


def _get_minimizer_default_method(
        method: Union[str, Callable,
                      None] = None) -> Union[str, Callable, None]:
    """Get the default method for the minimizer

    Examples
    --------
    >>> _get_minimizer_default_method(None)
    'broyden1'

    Parameters
    ----------
    method: str or None
        The method name

    Returns
    -------
    str
        The method name

    """
    if method is None:
        return "broyden1"
    else:
        return method
