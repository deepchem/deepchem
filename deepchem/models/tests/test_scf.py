import pytest
try:
    from deepchem.models.dft.scf import XCNNSCF
    import torch
    from deepchem.feat.dft_data import DFTEntry, DFTSystem
    from deepchem.models.dft.nnxc import HybridXC
    has_dqc = True
except ModuleNotFoundError:
    has_dqc = False
    pass


@pytest.mark.dqc
def construct_nn_model(ninp=2, nhid=10, ndepths=1):
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
    """
    layers = []
    for i in range(ndepths):
        n1 = ninp if i == 0 else nhid
        layers.append(torch.nn.Linear(n1, nhid))
        layers.append(torch.nn.Softplus())
    layers.append(torch.nn.Linear(nhid, 1, bias=False))
    return torch.nn.Sequential(*layers)


@pytest.mark.dqc
def test_scf():
    torch.manual_seed(42)
    nnmodel = construct_nn_model(ninp=2, nhid=10, ndepths=1).to(torch.double)
    hybridxc = HybridXC("lda_x", nnmodel, aweight0=0.0)
    e_type = 'ae'
    true_val = '0.09194410469'
    systems = [{
        'moldesc': 'Li 1.5070 0 0; H -1.5070 0 0',
        'basis': '6-311++G(3df,3pd)'
    }, {
        'moldesc': 'Li 0 0 0',
        'basis': '6-311++G(3df,3pd)',
        'spin': 1
    }, {
        'moldesc': 'H 0 0 0',
        'basis': '6-311++G(3df,3pd)',
        'spin': 1
    }]
    entry = DFTEntry.create(e_type, true_val, systems)
    evl = XCNNSCF(hybridxc, entry)
    system = DFTSystem(systems[1])
    run = evl.run(system)
    output = run.energy()
    expected_output = torch.tensor(-7.1914, dtype=torch.float64)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)
