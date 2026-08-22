import pytest
import numpy as np


@pytest.mark.torch
def test_pytorch_lda_x():
    """Testing the pytorch LDA XC functional."""
    from deepchem.utils.dft_utils.system.mol import Mol
    from deepchem.utils.dft_utils.qccalc.ks import KS
    from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchLDA

    # Test H2 System
    moldesc = "H 0 0 -0.74; H 0 0 0.74"
    basis = "sto-3g"
    system = Mol(moldesc=moldesc, basis=basis)

    # DeepChem LDA_X
    xc_dc = PyTorchLDA("lda_x")
    ks_dc = KS(system, xc_dc, variational=False)
    ks_dc.run()
    e_dc = ks_dc.energy().item()

    # Expected energy - examples/dft/dft_compare.py
    expected_e = -1.023727

    assert np.allclose(e_dc, expected_e)


@pytest.mark.torch
def test_pytorch_lda_get_edensityxc():
    import torch
    from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchLDA
    from deepchem.utils.dft_utils.data.datastruct import ValGrad

    xc = PyTorchLDA()
    n = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
    densinfo = ValGrad(value=n)
    edensity = xc.get_edensityxc(densinfo)
    assert edensity.shape == torch.Size([3])
