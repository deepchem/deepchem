import pytest
try:
    import torch
    import numpy as np
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_paulinet_electron_feature():
    from deepchem.models.torch_models.layers import PaulinetElectronFeature
    n_one = [10, 20]
    n_two = [30, 40]
    no_atoms = 5
    batch_size = 2
    total_electron = 10

    one_electron = torch.randn(batch_size, no_atoms, n_one[0])
    two_electron = torch.randn(batch_size, no_atoms, no_atoms, n_two[0])
    electron_feature_layer = PaulinetElectronFeature(n_one, n_two, no_atoms, batch_size, total_electron)
    one_electron_out, two_electron_out = electron_feature_layer(one_electron, two_electron)
    assert one_electron_out.shape == (batch_size, no_atoms, n_one[1])
    assert two_electron_out.shape == (batch_size, no_atoms, no_atoms, n_two[1])
