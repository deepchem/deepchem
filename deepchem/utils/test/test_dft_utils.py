import pytest
try:
    import torch
except:
    pass
import numpy as np


@pytest.mark.torch
def test_base_orb_params():
    from deepchem.utils.dft_utils import BaseOrbParams

    class MyOrbParams(BaseOrbParams):

        @staticmethod
        def params2orb(params, coeffs, with_penalty):
            return params, coeffs

        @staticmethod
        def orb2params(orb):
            return orb, torch.tensor([0], dtype=orb.dtype, device=orb.device)

    params = torch.randn(3, 4, 5)
    coeffs = torch.randn(3, 4, 5)
    with_penalty = 0.1
    orb, penalty = MyOrbParams.params2orb(params, coeffs, with_penalty)
    params2, coeffs2 = MyOrbParams.orb2params(orb)
    assert torch.allclose(params, params2)


@pytest.mark.torch
def test_qr_orb_params():
    from deepchem.utils.dft_utils import QROrbParams
    params = torch.randn(3, 3)
    coeffs = torch.randn(4, 3)
    with_penalty = 0.1
    orb, penalty = QROrbParams.params2orb(params, coeffs, with_penalty)
    params2, coeffs2 = QROrbParams.orb2params(orb)
    assert torch.allclose(orb, params2)


@pytest.mark.torch
def test_mat_exp_orb_params():
    from deepchem.utils.dft_utils import MatExpOrbParams
    params = torch.randn(3, 3)
    coeffs = torch.randn(4, 3)
    orb = MatExpOrbParams.params2orb(params, coeffs)
    params2, coeffs2 = MatExpOrbParams.orb2params(orb)
    assert coeffs2.shape == orb.shape


@pytest.mark.torch
def test_lattice():
    """Test lattice object.
    Comparing it's output with it's original implementation in dqc.
    """
    from deepchem.utils.dft_utils import Lattice
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    lattice = Lattice(a)

    assert torch.allclose(lattice.lattice_vectors(), a)
    assert torch.allclose(lattice.recip_vectors(),
                          torch.inverse(a.transpose(-2, -1)) * (2 * np.pi))
    assert torch.allclose(lattice.volume(), torch.det(a))

    assert torch.allclose(
        lattice.get_lattice_ls(1.0),
        torch.tensor([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.], [0., 0., 0.],
                      [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
    assert torch.allclose(
        lattice.get_gvgrids(6.0)[0],
        torch.tensor([[0.0000, 0.0000, -6.2832], [0.0000, -6.2832, 0.0000],
                      [-6.2832, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
                      [6.2832, 0.0000, 0.0000], [0.0000, 6.2832, 0.0000],
                      [0.0000, 0.0000, 6.2832]]))
    assert torch.allclose(
        lattice.get_gvgrids(6.0)[1],
        torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]))
    assert lattice.estimate_ewald_eta(1e-5) == 1.8
