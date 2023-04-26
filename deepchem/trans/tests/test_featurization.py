import deepchem as dc
from deepchem.molnet import load_delaney
from deepchem.trans.transformers import FeaturizationTransformer


def test_featurization_transformer():
    fp_size = 2048
    tasks, all_dataset, transformers = load_delaney('Raw')
    train = all_dataset[0]
    transformer = FeaturizationTransformer(
        dataset=train, featurizer=dc.feat.CircularFingerprint(size=fp_size))
    new_train = transformer.transform(train)

    assert new_train.y.shape == train.y.shape
    assert new_train.X.shape[-1] == fp_size
