try:
    from deepchem.data.data_loader import DFTYamlLoader
    has_dqc = True
except ModuleNotFoundError:
    has_dqc = False
import deepchem as dc
import pytest


@pytest.mark.dqc
def test_dftloader():
    input_files = 'deepchem/data/tests/dftdata.yaml'
    featurizer = dc.feat.DummyFeaturizer()
    k = DFTYamlLoader(featurizer)
    data = k.create_dftdataset(input_files, featurizer)
    assert ((data.X)[0]).get_true_val() == 0.09194410469
