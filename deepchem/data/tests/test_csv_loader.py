import os
import tempfile
import deepchem as dc
import numpy as np

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

def test_missing_labels_are_consistent_across_shard_sizes():
    """Missing labels should be masked identically regardless of shard size."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")

        with open(path, "w") as f:
            f.write(
                "smiles,label\n"
                "CCO,1.0\n"
                "CCC,\n"
                "CC,0.0\n"
                "CCCCC,1.0\n")

        loader = dc.data.CSVLoader(
            tasks=["label"],
            feature_field="smiles",
            featurizer=dc.feat.CircularFingerprint(size=8))

        unsharded = loader.create_dataset(
            [path],
            shard_size=None,
        )

        sharded = loader.create_dataset(
            [path],
            shard_size=2,
        )

        np.testing.assert_array_equal(
            unsharded.y,
            sharded.y,
        )

        np.testing.assert_array_equal(
            unsharded.w,
            sharded.w,
        )

        np.testing.assert_array_equal(
            unsharded.y.ravel(),
            np.array([1.0, 0.0, 0.0, 1.0]),
        )

        np.testing.assert_array_equal(
            unsharded.w.ravel(),
            np.array([1.0, 0.0, 1.0, 1.0]),
        )