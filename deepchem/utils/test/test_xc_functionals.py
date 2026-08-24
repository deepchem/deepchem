import pytest
import numpy as np


@pytest.mark.torch
def test_lda_x():
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
