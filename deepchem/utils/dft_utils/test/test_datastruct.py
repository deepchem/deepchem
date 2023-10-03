"""
Tests for DFT Datastructure Utilities
"""

try:
    import torch
    from deepchem.utils.dft_utils.datastruct import is_z_float, CGTOBasis
    has_torch = True
except:
    has_torch = False

def test_is_z_float():
    """Test is_z_float"""
    assert is_z_float(1.0) == True
    assert is_z_float(1) == False
    assert is_z_float(torch.tensor([1.0])) == True
    assert is_z_float(torch.tensor([1])) == False

def test_cgto_basis():
    """Test CGTOBasis"""
    expected_final_coeff = torch.tensor([204.8264]) # From Original Code
    basis = CGTOBasis(1, torch.tensor([30.0]), torch.tensor([15.0]))
    basis.wfnormalize_()
    assert torch.allclose(basis.coeffs, expected_final_coeff)
