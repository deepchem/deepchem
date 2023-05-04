from deepchem.data.data_loader import DFTYamlLoader
import pytest 
import torch 
from deepchem.models.losses import Loss, DensityProfileLoss

def test_densHF():
    inputs = 'deepchem/models/tests/assets/test_HFdp.yaml'
    data = DFTYamlLoader()
    dataset = data.create_dataset(inputs)
    labels = torch.as_tensor(dataset.y)
    nnmodel = (torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.Softplus(),
                                   torch.nn.Linear(10, 1, bias=False))).to(
                                       torch.double)
    hybridxc = HybridXC("lda_x", nnmodel, aweight0=0.0)
    entry = dataset.X[0]
    evl = XCNNSCF(hybridxc, entry)
    qcs = []
    for system in entry.get_systems():
        #syst = DFTSystem(system)
        qcs.append(evl.run(system))
    val = entry.get_val(qcs, entry) 
    output = torch.as_tensor(val)
    loss = (DensityProfileLoss()._create_pytorch_loss(volume))(output, labels)  
