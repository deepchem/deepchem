import warnings
from deepchem.models.torch_models.ChemCeption import ChemCeption

def test_chemception_forward_pass():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ChemCeption(augment=True)

        assert len(w)>0