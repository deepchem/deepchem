import pytest
try:
    import torch
except:
    pass


@pytest.mark.torch
def test_base_orb_params():
    from deepchem.models.dft.hamilton.orbparams import BaseOrbParams

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
    from deepchem.models.dft.hamilton.orbparams import QROrbParams
    params = torch.randn(3, 3)
    coeffs = torch.randn(4, 3)
    with_penalty = 0.1
    orb, penalty = QROrbParams.params2orb(params, coeffs, with_penalty)
    params2, coeffs2 = QROrbParams.orb2params(orb)
    assert torch.allclose(orb, params2)


@pytest.mark.torch
def test_mat_exp_orb_params():
    from deepchem.models.dft.hamilton.orbparams import MatExpOrbParams
    params = torch.randn(3, 3)
    coeffs = torch.randn(4, 3)
    orb = MatExpOrbParams.params2orb(params, coeffs)[0]
    params2, coeffs2 = MatExpOrbParams.orb2params(orb)
    assert coeffs2.shape == orb.shape
