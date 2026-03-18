import deepchem as dc
import numpy as np


def test_inmemory_features():
    smiles = ["C", "CC", "CCC", "CCCC"]
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)
    dataset = loader.create_dataset(smiles, shard_size=2)
    assert len(dataset) == 4
    assert dataset.X.shape == (4, 1024)
    assert dataset.get_number_shards() == 2
    assert (dataset.ids == np.arange(4)).all()


def test_inmemory_features_and_labels():
    smiles = ["C", "CC", "CCC", "CCCC"]
    labels = [1, 0, 1, 0]
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)
    dataset = loader.create_dataset(zip(smiles, labels), shard_size=2)
    assert len(dataset) == 4
    assert dataset.X.shape == (4, 1024)
    assert (dataset.y == np.array(labels)).all()
    assert dataset.get_number_shards() == 2
    assert (dataset.ids == np.arange(4)).all()


def test_inmemory_features_and_labels_and_weights():
    smiles = ["C", "CC", "CCC", "CCCC"]
    labels = [1, 0, 1, 0]
    weights = [1.5, 1.5, 1, 1]
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)
    dataset = loader.create_dataset(zip(smiles, labels, weights), shard_size=2)
    assert len(dataset) == 4
    assert dataset.X.shape == (4, 1024)
    assert (dataset.y == np.array(labels)).all()
    assert (dataset.w == np.array(weights)).all()
    assert (dataset.ids == np.arange(4)).all()
    assert dataset.get_number_shards() == 2


def test_inmemory_features_and_labels_and_weights_and_ids():
    smiles = ["C", "CC", "CCC", "CCCC"]
    labels = [1, 0, 1, 0]
    weights = [1.5, 1.5, 1, 1]
    ids = smiles
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.InMemoryLoader(tasks=["task1"], featurizer=featurizer)
    dataset = loader.create_dataset(zip(smiles, labels, weights, ids),
                                    shard_size=2)
    assert len(dataset) == 4
    assert dataset.X.shape == (4, 1024)
    assert (dataset.y == np.array(labels)).all()
    assert (dataset.w == np.array(weights)).all()
    assert (dataset.ids == np.array(ids)).all()
    assert dataset.get_number_shards() == 2
