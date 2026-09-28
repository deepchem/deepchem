import os
import tempfile
import deepchem as dc


def test_load_singleton_csv():
    fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fin.write("smiles,endpoint\nc1ccccc1,1")
    fin.close()
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["endpoint"]
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    X = loader.create_dataset(fin.name)
    assert len(X) == 1
    os.remove(fin.name)


def _missing_label_csv():
    """A CSV whose second row has no label, which must not become a NaN target."""
    fin = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    fin.write("smiles,endpoint\nc1ccccc1,1\nCCCCCC,\nc1ccncc1,0\n")
    fin.close()
    return fin.name


def test_missing_label_is_masked_without_a_shard_size():
    """A missing label must get weight 0, not weight 1 against a NaN target.

    The default path passes the whole file in one frame, so a NaN left in the
    label column is what the model would be trained on. create_labels_and_weights
    only recognises a missing label as the empty string, and that check is
    restricted to an object/unicode column, so a float column silently escapes
    the mask.
    """
    path = _missing_label_csv()
    try:
        loader = dc.data.CSVLoader(
            tasks=["endpoint"],
            feature_field="smiles",
            featurizer=dc.feat.CircularFingerprint(size=1024))
        dataset = loader.create_dataset(path, shard_size=None)
        y = dataset.y.ravel()
        w = dataset.w.ravel()
        assert not any(v != v for v in y), "a NaN label survived: {}".format(y)
        assert list(w) == [1.0, 0.0, 1.0]
    finally:
        os.remove(path)


def test_sharded_and_unsharded_agree_on_a_missing_label():
    """Shard size is a performance knob, so it must not change the training set."""
    path = _missing_label_csv()
    try:
        loader = dc.data.CSVLoader(
            tasks=["endpoint"],
            feature_field="smiles",
            featurizer=dc.feat.CircularFingerprint(size=1024))
        whole = loader.create_dataset(path, shard_size=None)
        sharded = loader.create_dataset(path, shard_size=1)
        assert list(whole.w.ravel()) == list(sharded.w.ravel())
        assert list(whole.y.ravel()) == list(sharded.y.ravel())
    finally:
        os.remove(path)
