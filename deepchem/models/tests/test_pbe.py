import pytest
import warnings
try:
    import torch
except Exception as e:
    warnings.warn("Could not import torch. Skipping tests." + str(e))


@pytest.mark.dqc
def test_pbe():
    from deepchem.models.dft.nnxc import HybridXC
    from deepchem.models.dft.dftxc import _construct_nn_model
    from deepchem.models.dft.scf import XCNNSCF
    from deepchem.feat.dft_data import DFTEntry
    input_size = 3
    hidden_size = 3
    n_layers = 3
    modeltype = 1
    nnmodel = _construct_nn_model(input_size, hidden_size, n_layers,
                                  modeltype).to(torch.double)
    e_type = 'ae'
    true_val = 0.237898
    systems = [{
        'moldesc': 'Be 0 0 0; H -2.5065 0 0; H 2.5065 0 0',
        'basis': '6-311++G(3df,3pd)'
    }, {
        'moldesc': 'H 0 0 0',
        'basis': '6-311++G(3df,3pd)',
        'spin': '1',
        'number': '2'
    }, {
        'moldesc': 'Be 0 0 0',
        'basis': '6-311++G(3df,3pd)'
    }]

    entry = DFTEntry.create(e_type, true_val, systems)
    hybridxc = HybridXC("gga_x_pbe", nnmodel, aweight0=0.0)
    evl = XCNNSCF(hybridxc, entry)
    qcs = []
    for system in entry.get_systems():
        qcs.append(evl.run(system))
    output = qcs[0].energy()
    expected_output = torch.tensor(-15.7262, dtype=torch.float64)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)
