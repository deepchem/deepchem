import pytest
import warnings
import numpy as np
try:
    import torch
except Exception as e:
    warnings.warn("Could not import torch. Skipping tests. Error is: " + str(e))


@pytest.mark.dqc
def test_densHF():
    from deepchem.data.data_loader import DFTYamlLoader
    from deepchem.models.dft.scf import XCNNSCF
    from deepchem.models.dft.nnxc import HybridXC
    from deepchem.models.losses import DensityProfileLoss
    inputs = 'deepchem/models/tests/assets/test_HFdp.yaml'
    data = DFTYamlLoader()
    dataset = data.create_dataset(inputs)
    labels = torch.as_tensor(dataset.y)
    nnmodel = (torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.Softplus(),
                                   torch.nn.Linear(10, 1, bias=False))).to(
                                       torch.double)
    hybridxc = HybridXC("lda_x", nnmodel, aweight0=0.0)
    entry = dataset.X[0]
    grid = (dataset.X[0]).get_integration_grid()
    volume = grid.get_dvolume()
    evl = XCNNSCF(hybridxc, entry)
    qcs = []
    for system in entry.get_systems():
        qcs.append(evl.run(system))
    val = entry.get_val(qcs)
    output = torch.as_tensor(val)
    loss = ((DensityProfileLoss()._create_pytorch_loss(volume))(
        output, labels)).detach().numpy()
    expected = np.array(0.0068712)
    assert np.allclose(loss, expected)
