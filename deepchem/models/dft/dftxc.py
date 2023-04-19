"""Derived from https://github.com/mfkasim1/xcnn/blob/f2cb9777da2961ac553f256ecdcca3e314a538ca/xcdnn2/litmodule.py"""
from deepchem.models.dft.scf import XCNNSCF
import torch
from deepchem.models.dft.nnxc import HybridXC
from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel
from typing import Optional
import numpy as np


class DFTXC(torch.nn.Module):
    """
    This layer initializes the neural network exchange correlation functional and
    the hybrid functional. It is then used to run the Kohn Sham iterations.
    """

    def __init__(self,
                 xcstr: str,
                 ninp: int = 2,
                 nhid: int = 10,
                 ndepths: int = 1,
                 modeltype: int = 1):
        """
        Parameters
        ----------
        xcstr: str
            The choice of xc to use. Some of the commonly used ones are:
            lda_x, lda_c_pw, lda_c_ow, lda_c_pz, lda_xc_lp_a, lda_xc_lp_b.
        ninp: int
        nhid: int
        ndepths: int
        modeltype: int
        """
        super(DFTXC, self).__init__()
        self.xcstr = xcstr
        self.model = _construct_nn_model(ninp, nhid, ndepths,
                                         modeltype).to(torch.double)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: list of tensors containing dataset

        Returns
        -------
        torch.Tensor
            Calculated value of the data point after running the Kohn Sham iterations
            using the neural network XC functional.
        """
        hybridxc = HybridXC(self.xcstr, self.model, aweight0=0.0)
        for entry in inputs:
            evl = XCNNSCF(hybridxc, entry)
            qcs = []
            for system in entry.get_systems():
                qcs.append(evl.run(system))
            if entry.entry_type == 'dm':
                return torch.as_tensor(entry.get_val(qcs)[0])
            else:
                return torch.as_tensor(entry.get_val(qcs))


class XCModel(TorchModel):
    """
    This class is used to initialize and run Differentiable Quantum Chemistry (i.e,
    DFT) calculations, using an exchange correlation functional that has been replaced
    by a neural network. This model is based on the paper "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." and is listed below for reference.

    To read more about Density Functional Theory and the exchange
    correlation functional please check the references below.

    Examples
    --------
    >>> from deepchem.models.dft.dftxc import XCModel
    >>> from deepchem.data.data_loader import DFTYamlLoader
    >>> inputs = 'deepchem/models/tests/assets/test_dftxcdata.yaml'
    >>> data = DFTYamlLoader()
    >>> dataset = (data.create_dataset(inputs))
    >>> model = XCModel("lda_x", batch_size=1)
    >>> loss = model.fit(dataset, nb_epoch=1, checkpoint_interval=1)

    Notes
    -----
    The entry type "Density Matrix" cannot be used on model.evaluate as of now.
    To run predictions on this data type, a dataset containing only "dm" entries must
    be used.

    References
    ----------
    deepchem.models.dft.nnxc
    Encyclopedia of Condensed Matter Physics, 2005.
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." Physical Review Letters 127.12 (2021): 126403.
    """

    def __init__(self,
                 xcstr: str,
                 ninp: int = 2,
                 nhid: int = 10,
                 ndepths: int = 1,
                 modeltype: int = 1,
                 n_tasks: int = 0,
                 log_frequency: int = 0,
                 mode: str = 'classification',
                 device: Optional[torch.device] = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        xcstr: str
            The choice of xc to use.
        ninp: int
            size of neural input
        nhid: int
            hidden layer size
        ndepths: int
            depth of neural network
        modeltype: int
            model type 2 includes an activation layer whereas type 1 does not.
        """

        model = DFTXC(xcstr, ninp, nhid, ndepths, modeltype)
        self.xc = xcstr
        self.model = model
        loss: Loss = L2Loss()
        output_types = ['loss', 'predict']
        self.mode = mode
        super(XCModel, self).__init__(model,
                                      loss=loss,
                                      output_types=output_types,
                                      **kwargs)

    def _prepare_batch(self, batch):

        inputs, labels, weights = batch
        labels = [torch.from_numpy(inputs[0][0].get_true_val())]
        labels[0].requires_grad_()
        w = np.array([1.0])
        weights = [torch.from_numpy(w)]
        return (inputs, labels, weights)


class ExpM1Activation(torch.nn.Module):
    """
    This class is an activation layer that is used with model_type 2.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - 1


def _construct_nn_model(ninp: int, nhid: int, ndepths: int, modeltype: int):
    """
    Constructs Neural Network

    Parameters
    ----------
    ninp: int
        size of neural input
    nhid: int
        hidden layer size
    ndepths: int
        depth of neural network
    modeltype: int
        model type 2 includes an activation layer whereas type 1 does not.

    Returns
    -------
    torch.nn.Sequential(*layers)
    """
    if modeltype == 1:
        layers = []
        for i in range(ndepths):
            n1 = ninp if i == 0 else nhid
            layers.append(torch.nn.Linear(n1, nhid))
            layers.append(torch.nn.Softplus())
        layers.append(torch.nn.Linear(nhid, 1, bias=False))
        return torch.nn.Sequential(*layers)
    elif modeltype == 2:
        layers = []
        for i in range(ndepths):
            n1 = ninp if i == 0 else nhid
            layers.append(torch.nn.Linear(n1, nhid))
            if i < ndepths - 1:
                layers.append(torch.nn.Softplus())
            else:
                layers.append(ExpM1Activation())
        layers.append(torch.nn.Linear(nhid, 1, bias=False))
        return torch.nn.Sequential(*layers)
