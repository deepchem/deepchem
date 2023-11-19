import torch
import warnings
from typing import Callable, List, Any, Union, Sequence
from deepchem.utils.differentiation_utils import assert_type, LinearOperator, \
    get_pure_function, make_sibling, PureFunction, TensorNonTensorSeparator

__all__ = ["jac", "hess"]

def jac(fcn: Callable[..., torch.Tensor], params: Sequence[Any],
        idxs: Union[None, int, Sequence[int]] = None) -> Union[LinearOperator, List]:
    """
    Returns the LinearOperator that acts as the jacobian of the params.
    The shape of LinearOperator is (nout, nin) where `nout` and `nin` are the
    total number of elements in the output and the input, respectively.

    Arguments
    ---------
    * fcn: Callable[...,torch.Tensor]
        Callable with tensor output and arbitrary numbers of input parameters.
    * params: Sequence[Any]
        List of input parameters of the function.
    * idxs: int or list of int or None
        List of the parameters indices to get the jacobian.
        The pointed parameters in `params` must be tensors and requires_grad.
        If it is None, then it will return all jacobian for all parameters that
        are tensor which requires_grad.

    Returns
    -------
    * linops: list of LinearOperator
        List of LinearOperator of the jacobian
    """
    # check idxs
    idxs_list = _setup_idxs(idxs, params)

    # make the function a functional (depends on all parameters in the object)
    pfcn = get_pure_function(fcn)
    res = [_Jac(pfcn, params, idx) for idx in idxs_list]
    if isinstance(idxs, int):
        return res[0]
    return res

def hess(fcn: Callable[..., torch.Tensor], params: Sequence[Any],
         idxs: Union[None, int, Sequence[int]] = None) -> Union[LinearOperator, List]:
    """
    Returns the LinearOperator that acts as the Hessian of the params.
    The shape of LinearOperator is (nin, nin) where `nin` is the
    total number of elements in the input.

    Arguments
    ---------
    * fcn: Callable[...,torch.Tensor]
        Callable with tensor output and arbitrary numbers of input parameters.
        The numel of the output must be 1.
    * params: Sequence[Any]
        List of input parameters of the function.
    * idxs: int or list of int or None
        List of the parameters indices to get the jacobian.
        The pointed parameters in `params` must be tensors and requires_grad.
        If it is None, then it will return all Hessian for all parameters that
        are tensor which requires_grad.

    Returns
    -------
    * linops: list of LinearOperator
        List of LinearOperator of the Hessian
    """
    idxs_list = _setup_idxs(idxs, params)

    # make the function a functional (depends on all parameters in the object)
    pfcn = get_pure_function(fcn)

    res = []

    def gen_pfcn2(idx):
        @make_sibling(pfcn)
        def pfcn2(*params):
            with torch.enable_grad():
                z = pfcn(*params)
            grady, = torch.autograd.grad(z, (params[idx],), retain_graph=True,
                                         create_graph=torch.is_grad_enabled())
            return grady
        return pfcn2

    for idx in idxs_list:
        # suppress warnings of double implementation in hermitian matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hs = _Jac(gen_pfcn2(idx), params, idx, is_hermitian=True)
        res.append(hs)

    if isinstance(idxs, int):
        return res[0]
    return res

class _Jac(LinearOperator):
    def __init__(self, fcn: PureFunction,
                 params: Sequence[Any], idx: int, is_hermitian=False) -> None:

        # TODO: check if fcn has kwargs

        # run once to get the shapes and numels
        yparam = params[idx]
        with torch.enable_grad():
            yout = fcn(*params)  # (*nout)
            v = torch.ones_like(yout).to(yout.device).requires_grad_()  # (*nout)
            dfdy, = torch.autograd.grad(yout, (yparam,), grad_outputs=v, create_graph=True)  # (*nin)

        inshape = yparam.shape
        outshape = yout.shape
        nin = torch.numel(yparam)
        nout = torch.numel(yout)

        super(_Jac, self).__init__(
            shape=(nout, nin),
            is_hermitian=is_hermitian,
            dtype=yparam.dtype,
            device=yparam.device)

        self.fcn = fcn
        self.yparam = yparam
        self.params = list(params)
        self.objparams = fcn.objparams()
        self.yout = yout
        self.v = v
        self.idx = idx
        self.dfdy = dfdy
        self.inshape = inshape
        self.outshape = outshape
        self.nin = nin
        self.nout = nout

        # params tensor is the LinearOperator's parameters
        self.param_sep = TensorNonTensorSeparator(params)
        self.params_tensor = self.param_sep.get_tensor_params()
        self.id_params_tensor = [id(param) for param in self.params_tensor]
        self.id_objparams_tensor = [id(param) for param in self.objparams]

    def _getparamnames(self, prefix: str = "") -> List[str]:
        return [prefix + "yparam"] + \
               [prefix + ("params_tensor[%d]" % i) for i in range(len(self.params_tensor))] + \
               [prefix + ("objparams[%d]" % i) for i in range(len(self.objparams))]

    def _mv(self, gy: torch.Tensor) -> torch.Tensor:
        # gy: (..., nin)
        # returns: (..., nout)

        # if the object parameter is still the same, then use the pre-calculated values
        if self.__param_tensors_unchanged():
            v = self.v
            dfdy = self.dfdy
        # otherwise, reevaluate by replacing the parameters with the new tensor params
        else:
            with torch.enable_grad(), self.fcn.useobjparams(self.objparams):
                self.__update_params()
                yparam = self.params[self.idx]
                yout = self.fcn(*self.params)  # (*nout)
                v = torch.ones_like(yout).to(yout.device).requires_grad_()  # (*nout)
                dfdy, = torch.autograd.grad(yout, (yparam,), grad_outputs=v, create_graph=True)  # (*nin)

        gy1 = gy.reshape(-1, self.nin)  # (nbatch, nin)
        nbatch = gy1.shape[0]
        dfdyfs_list = []
        for i in range(nbatch):
            dfdyf, = torch.autograd.grad(dfdy, (v,), grad_outputs=gy1[i].reshape(self.inshape),
                                         retain_graph=True, create_graph=torch.is_grad_enabled())  # (*nout)
            dfdyfs_list.append(dfdyf.unsqueeze(0))
        dfdyfs = torch.cat(dfdyfs_list, dim=0)  # (nbatch, *nout)

        res = dfdyfs.reshape(*gy.shape[:-1], self.nout)  # (..., nout)
        res = connect_graph(res, self.params_tensor)
        res = connect_graph(res, self.objparams)
        return res

    def _rmv(self, gout: torch.Tensor) -> torch.Tensor:
        # gout: (..., nout)
        # self.yfcn: (*nin)
        if self.__param_tensors_unchanged():
            yout = self.yout
            yparam = self.yparam
        else:
            with torch.enable_grad(), self.fcn.useobjparams(self.objparams):
                self.__update_params()
                yparam = self.params[self.idx]
                yout = self.fcn(*self.params)  # (*nout)

        gout1 = gout.reshape(-1, self.nout)  # (nbatch, nout)
        nbatch = gout1.shape[0]
        dfdy_list = []
        for i in range(nbatch):
            one_dfdy, = torch.autograd.grad(yout, (yparam,), grad_outputs=gout1[i].reshape(self.outshape),
                                            retain_graph=True, create_graph=torch.is_grad_enabled())  # (*nin)
            dfdy_list.append(one_dfdy.unsqueeze(0))
        dfdy = torch.cat(dfdy_list, dim=0)  # (nbatch, *nin)

        res = dfdy.reshape(*gout.shape[:-1], self.nin)  # (..., nin)
        res = connect_graph(res, self.params_tensor)
        res = connect_graph(res, self.objparams)
        return res  # (..., nin)

    def __param_tensors_unchanged(self):
        return [id(param) for param in self.params_tensor] == self.id_params_tensor and \
               [id(param) for param in self.objparams] == self.id_objparams_tensor

    def __update_params(self):
        self.params = self.param_sep.reconstruct_params(self.params_tensor)

def connect_graph(out, params):
    # just to have a dummy graph, in case there is a parameter that
    # is disconnected in calculating df/dy
    return out + sum([p.reshape(-1)[0] * 0 for p in params])

def _setup_idxs(idxs, params):
    if idxs is None:
        idxs = [i for i, t in enumerate(params) if isinstance(t, torch.Tensor) and t.requires_grad]
    elif isinstance(idxs, int):
        idxs = [idxs]

    for p in idxs:
        assert_type(isinstance(params[p], torch.Tensor) and params[p].requires_grad,
                    "The %d-th element (0-based) must be a tensor which requires grad" % p)
    return idxs
