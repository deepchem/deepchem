from deepchem.data.data_loader import DFTYamlLoader
import deepchem as dc


def test_dftloader():
    input_files = 'dftdata.yaml'
    featurizer = dc.feat.DummyFeaturizer()
    k = DFTYamlLoader(featurizer)
    l = k.create_dataset(input_files, featurizer)
    assert ((l.X)[0]).get_true_val() == 0.09194410469
