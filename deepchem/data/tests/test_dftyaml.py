try:
    from deepchem.data.data_loader import DFTYamlLoader
    has_dqc = True
except ModuleNotFoundError:
    has_dqc = False
import pytest


@pytest.mark.dqc
def test_dftloader():
    inputs = ['deepchem/data/tests/dftdata.yaml']
    k = DFTYamlLoader()
    data = k.create_dataset(inputs)
    assert data.X.dtype == ('O')
    assert len(data) == 2
    assert ((data.X)[0]).get_weight() == 1340
    assert ((data.X)[0]).get_true_val() == 0.09194410469


@pytest.mark.dqc
def test_dftshard():
    data = DFTYamlLoader()
    dataset = data.create_dataset([
        'deepchem/data/tests/dftdata.yaml',
        'deepchem/models/tests/assets/test_dftxcdata.yaml'
    ],
                                  shard_size=4)
    check_val = 0.09194410469
    assert dataset.y[2][0] == check_val
