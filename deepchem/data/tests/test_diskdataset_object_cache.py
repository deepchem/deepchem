import numpy as np
import pytest
from deepchem.data import DiskDataset

def test_object_array_skips_cache(tmp_path):
    X = np.array(["a", "b", "c"], dtype=object)
    y = None
    w = None
    ids = np.array([0, 1, 2])

    dataset = DiskDataset.create_dataset(
        [(X, y, w, ids)],
        data_dir=tmp_path
    )

    with pytest.warns(UserWarning):
        shard = dataset.get_shard(0)

    assert shard[0].dtype == object
