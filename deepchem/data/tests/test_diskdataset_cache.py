import pytest

pytest.importorskip("rdkit")

import numpy as np
from deepchem.data.datasets import DiskDataset


def test_diskdataset_disable_cache():
    """Test that setting memory_cache_size=0 disables DiskDataset caching."""

    X = np.array([{"a": i} for i in range(3)], dtype=object)
    dataset = DiskDataset.from_numpy(X)

    dataset.memory_cache_size = 0
    dataset.get_shard(0)

    assert dataset._cached_shards is None
