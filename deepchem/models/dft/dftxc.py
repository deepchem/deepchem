from deepchem.data.data_loader import DFTYamlLoader
from deepchem.models.dft.scf import XCNNSCF
import torch
from deepchem.feat.dft_data import DFTEntry, DFTSystem
from deepchem.models.dft.nnxc import HybridXC
from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel


class DFTXC(torch.nn.module):

    def __init__(self, nnmodel, input):

        load_data = DFTYamlLoader()
        self.data = load_data.create_dataset(input)
        nnmodel = model
        self.hybridxc = HybridXC("lda_x", nnmodel, aweight0=0.0)

    def inputs(self):
        return self.data

    def forward(self):
        for i in (self.data).X:
            entry = i[0]
            evl = XCNNSCF(self.hybridxc, entry)
            for system in entry.get_systems():
                qcs = [evl.run(system)]
            return entry.get_val(qcs)


class XCModel(TorchModel):

    def __init__(self,
                 input: str,
                 nnmodel: Optional[torch.nn.Module] = None,
                 device: Optional[torch.device] = None,
                 **kwargs) -> None:
        if nnmodel == None:
            nnmodel = construct_nn_model(ninp, nhid, ndepths,
                                         modeltype).to(torch.double)
        model = DFTXC(nnmodel, input) = self.model
        loss: Loss = L2Loss()
        output_types = ['loss']
        super(XCModel, self).__init__(model,
                                      loss=loss,
                                      output_types=output_types,
                                      **kwargs)

        def _prepare_batch(self, batch):
            inputs = (self.model).inputs()
            labels = [
                torch.from_numpy(input.get_true_val()) for input in inputs
            ]
            return inputs, labels


class ExpM1Activation(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - 1


def construct_nn_model(ninp: int, nhid: int, ndepths: int, modeltype: int):
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
