from abc import abstractproperty, abstractmethod
from typing import Union
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam
from dqc.api.getxc import get_xc
import torch


class BaseNNXC(BaseXC, torch.nn.Module):
    """
    Base class for the NNLDA and HybridXC classes.
    """

    @abstractproperty
    def family(self) -> int:
        """
        This method determines the type of model to be used, to train the 
        neural network. Currently we only support an LDA based model and will 
        implement more in subsequent iterations.

        Returns
        -------
        xc.family 
        """  
        pass

    @abstractmethod
    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """
        This method is used to transform the electron density. The output
        of this method varies depending on the layer. 

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information calculated using DQC utilities.
        """
        pass


class NNLDA(BaseNNXC):
    """
    Neural network xc functional of LDA (only receives the density as input)

    """

    def __init__(self,
                 nnmodel: torch.nn.Module,
                 ninpmode: int = 1,
                 outmultmode: int = 1):
        super().__init__()
        """
        Parameters
        ----------
        nnmodel: torch.nn.Module
            Neural network for xc functional
        ninpmode: int
            The mode to decide the transformation of the density to NN input.
        outmultmode: int
            The mode to decide Eks from NN output
        """
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.outmultmode = outmultmode

    @property
    def family(self) -> int:
        return 1

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """
        This method transform the local electron density (n) and the spin
        density (xi) for polarized and unpolarized cases, to be the input of
        the neural network.

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information calculated using DQC utilities.
        Returns
        -------
        res
            Neural network output by calculating total density (n) and the spin density (xi)
        """
        if isinstance(densinfo, ValGrad):  # unpolarized case
            n = densinfo.value.unsqueeze(-1)  # (*BD, nr, 1)
            xi = torch.zeros_like(n)
        else:  # polarized case
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd  # (*BD, nr, 1)
            xi = (nu - nd) / (n + 1e-18)  # avoiding nan

        ninp = n

        x = torch.cat((ninp, xi), dim=-1)  # (*BD, nr, 2)
        nnout = self.nnmodel(x)  # (*BD, nr, 1)
        res = nnout * n  # (*BD, nr, 1)
        res = res.squeeze(-1)
        return res


class HybridXC(BaseNNXC):
    """
    The HybridXC module computes XC energy by summing XC energy computed
    from libxc and the trainable neural network with tunable weights.
    """

    def __init__(self,
                 xcstr: str,
                 nnmodel: torch.nn.Module,
                 ninpmode: int = 1,
                 outmultmode: int = 1,
                 aweight0: float = 0.0,
                 bweight0: float = 1.0):

        super().__init__()
        """
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
        aweight0: float
            weight of the neural network
         bweight0: float
            weight of the default xc
        """
        self.xc = get_xc(xcstr)
        if self.xc.family == 1:
            self.nnxc = NNLDA(nnmodel,
                              ninpmode=ninpmode,
                              outmultmode=outmultmode)
        self.aweight = torch.nn.Parameter(
            torch.tensor(aweight0, requires_grad=True))
        self.bweight = torch.nn.Parameter(
            torch.tensor(bweight0, requires_grad=True))
        self.weight_activation = torch.nn.Identity()

    @property
    def family(self) -> int:
        return self.xc.family

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Get electron density from xc
        This function reflects eqn. 4 in the `paper <https://arxiv.org/abs/2102.04229>_`.
        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information calculated using DQC utilities.

        Returns
        -------
        Total calculated electron density with tunable weights.
        """
        nnlda_ene = self.nnxc.get_edensityxc(densinfo)
        lda_ene = self.xc.get_edensityxc(densinfo)
        aweight = self.weight_activation(self.aweight)
        bweight = self.weight_activation(self.bweight)
        return nnlda_ene * aweight + lda_ene * bweight
