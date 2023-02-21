from abc import abstractproperty, abstractmethod
import numpy as np
from typing import Union, Iterator, List
try:
    from dqc.xc.base_xc import BaseXC
    from dqc.utils.datastruct import ValGrad, SpinParam
    from dqc.utils.safeops import safenorm, safepow
    from dqc.api.getxc import get_xc
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError("This layer requires dqc and torch")

class BaseNNXC(BaseXC, torch.nn.Module):
    """
    Base class for the NNLDA and HybridXC classes.
    """
    @abstractproperty
    def family(self) -> int:
        pass

    @abstractmethod
    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        pass

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        # torch.nn.module prefix has no ending dot, while xt prefix has
        nnprefix = prefix if prefix == "" else prefix[:-1]
        return [
            name for (name, param) in self.named_parameters(prefix=nnprefix)
        ]

class NNLDA(BaseNNXC):
    """
    Neural network xc functional for LDA
    neural network xc functional of LDA (only receives the density as input)
    Parameters
    ----------
    nnmodel: torch.nn.Module
        Neural network for xc functional
    ninpmode: int
        The mode to decide the transformation of the density to NN input.
    outmultmode: int
        The mode to decide Eks from NN output
    """

    def __init__(self,
                 nnmodel: torch.nn.Module,
                 ninpmode: int = 1,
                 outmultmode: int = 1):
        super().__init__()
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.outmultmode = outmultmode

    @property
    def family(self) -> int:
        return 1

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # collect the total density (n) and the spin density (xi)
        if isinstance(densinfo, ValGrad):  # unpolarized case
            n = densinfo.value.unsqueeze(-1)  # (*BD, nr, 1)
            xi = torch.zeros_like(n)
        else:  # polarized case
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd  # (*BD, nr, 1)
            xi = (nu - nd) / (n + 1e-18)  # avoiding nan

        # decide how to transform the density to be the input of nn
        ninp = n

        # get the neural network output
        x = torch.cat((ninp, xi), dim=-1)  # (*BD, nr, 2)
        nnout = self.nnmodel(x)  # (*BD, nr, 1)
        res = nnout * n  # (*BD, nr, 1)
        res = res.squeeze(-1)
        return res


class HybridXC(BaseNNXC):
    """
    The HybridXC module computes XC energy by summing XC energy computed
    from libxc and the trainable neural network with tunable weights.
    Parameters
    ----------
    xcstr: str
        The choice of xc to use.
    nnmodel: nn.Module
        trainable neural network for prediction xc energy. 
    ninpmode: int
        The mode to decide the transformation of the density to NN input.
    outmultmode: int
        The mode to decide Eks from NN output
    """

    # default value of xcstr is lda_x
    def __init__(
            self,
            xcstr: str,
            nnmodel: torch.nn.Module,
            *,
            ninpmode:
        int = 1,  # mode to decide how to transform the density to nn input
            outmultmode: int = 1,  # mode of calculating Eks from output of nn
            aweight0: float = 0.0,  # weight of the neural network
            bweight0: float = 1.0,  # weight of the default xc
            dtype: torch.dtype = torch.double):
        # hybrid libxc and neural network xc where it starts as libxc and then
        # trains the weights of libxc and nn xc

        super().__init__()
        # What is type of xc here?
        self.xc = get_xc(xcstr)
        if self.xc.family == 1:
            self.nnxc = NNLDA(nnmodel,
                              ninpmode=ninpmode,
                              outmultmode=outmultmode)
        self.aweight = torch.nn.Parameter(
            torch.tensor(aweight0,
                         dtype=dtype,
                         requires_grad=True))
        self.bweight = torch.nn.Parameter(
            torch.tensor(bweight0,
                         dtype=dtype,
                         requires_grad=True))
        self.weight_activation = torch.nn.Identity()

    @property
    def family(self) -> int:
        return self.xc.family

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Get electron density from xc
        This function reflects eqn. 4 in the `paper <https://arxiv.org/abs/2102.04229>_`.
        """
        nnlda_ene = self.nnxc.get_edensityxc(densinfo)
        lda_ene = self.xc.get_edensityxc(densinfo)
        aweight = self.weight_activation(self.aweight)
        bweight = self.weight_activation(self.bweight)
        return nnlda_ene * aweight + lda_ene * bweight
