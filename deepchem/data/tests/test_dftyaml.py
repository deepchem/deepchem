try:
    from deepchem.data.data_loader import DFTYamlLoader
    has_dqc = True
except ModuleNotFoundError:
    has_dqc = False
import deepchem as dc
import pytest


@pytest.mark.dqc
def test_dftloader():
    inputs = 'deepchem/data/tests/dftdata.yaml'
    k = DFTYamlLoader()
    data_dir = 'deepchem/data'
    data = k.create_dataset(inputs, data_dir)
    assert ((data.X)[0][0]).get_true_val() == 0.09194410469
