import pytest
import deepchem as dc
import tempfile
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass

@pytest.mark.torch
def test_wln_scoring_function_construction():
    """
    Test if WLNScoringFunction can be constructed without any errors.
    """
    from deepchem.models.torch_models.wln import WLNScoring
    scoring_function = WLNScoring(
        atom_feature_dim=32,
        wln_bond_fdim=6,
        binary_fdim=64,
        hidden_size=128,
        depth=3,
        num_bond_orders=5
    )
    assert scoring_function is not None
    
@pytest.mark.torch
def test_output_shape():
    """
    Test if WLNScoringFunction produces output of expected shape.
    """
    from deepchem.models.torch_models.wln import WLN

    batch_size = 2
    max_atoms = 5
    atom_feature_dim = 32
    wln_bond_fdim = 6
    binary_fdim = 64

    scoring_function = WLN(
        atom_feature_dim=atom_feature_dim,
        wln_bond_fdim=wln_bond_fdim,
        binary_fdim=binary_fdim,
        hidden_size=128,
        depth=3,
        num_bond_orders=5
    )

    atom_features = torch.randn(batch_size, max_atoms, atom_feature_dim)
    adj_matrix = torch.ones(batch_size, max_atoms, max_atoms)
    wln_bond_features = torch.randn(batch_size, max_atoms, max_atoms, wln_bond_fdim)
    binary_features = torch.randn(batch_size, max_atoms, max_atoms, binary_fdim)
    atom_mask = torch.ones(batch_size, max_atoms)

    scores = scoring_function(
        atom_features,
        adj_matrix,
        wln_bond_features,
        binary_features,
        atom_mask
    )

    assert scores.shape == (batch_size, max_atoms, max_atoms, 5)
    

