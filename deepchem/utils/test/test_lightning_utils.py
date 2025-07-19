import numpy as np
import pytest
try:
    from torch.utils.data import DataLoader
    from deepchem.data import DiskDataset, _TorchIndexDiskDataset
except ImportError as e:
    pytest.skip(f"Skipping tests due to missing dependencies: {e}",
                allow_module_level=True)
from deepchem.utils.lightning_utils import collate_dataset_fn


@pytest.fixture(scope="module")
def dummy_disk_dataset(tmp_path_factory):
    """
    Creates a DiskDataset with multiple, unevenly sized shards for testing.

    This fixture is scoped to the module, so the dataset is only created
    once per test file, making tests run faster.

    The dataset will have:
    - 3 shards with sizes [5, 3, 4]
    - Total samples: 12
    - Features (X): 2 per sample, with values from 0 to 23
    - Labels (y): 1 per sample, with values from 0 to 11
    - IDs: "id_0", "id_1", ..., "id_11"
    """
    data_dir = tmp_path_factory.mktemp("getitem_dataset")
    shard_sizes = [5, 3, 4]
    n_samples = sum(shard_sizes)
    n_features = 2
    n_tasks = 1

    # Create the full dataset in memory first
    X = np.arange(n_samples * n_features,
                  dtype=np.float32).reshape(n_samples, n_features)
    y = np.arange(n_samples, dtype=np.float32).reshape(n_samples, n_tasks)
    w = np.ones((n_samples, n_tasks), dtype=np.float32)
    ids = np.array([f"id_{i}" for i in range(n_samples)])

    # Create a generator that yields the data in shards
    def shard_generator():
        start = 0
        for size in shard_sizes:
            end = start + size
            yield (X[start:end], y[start:end], w[start:end], ids[start:end])
            start = end

    # Use the generator to create the DiskDataset and wrap it in IndexDatasetWrapper
    dataset = _TorchIndexDiskDataset(
        DiskDataset.create_dataset(shard_generator(), data_dir=data_dir))
    return dataset


@pytest.mark.torch
def test_cumulative_sum(dummy_disk_dataset):
    """
    Tests the internal _cumulative_sum method to ensure it calculates correctly.
    """
    # Shard sizes are [0, 5, 3, 4]
    expected_sums = [0, 5, 8, 12]
    # Access the private method for testing purposes
    calculated_sums = dummy_disk_dataset._cumulative_sum()
    assert calculated_sums == expected_sums


@pytest.mark.torch
def test_getitem_first_and_last(dummy_disk_dataset):
    """Tests retrieving the very first and very last elements."""
    # --- Test first element (index 0) ---
    x0, y0, w0, id0 = dummy_disk_dataset[0]
    np.testing.assert_array_equal(x0, np.array([0., 1.], dtype=np.float32))
    np.testing.assert_array_equal(y0, np.array([0.], dtype=np.float32))
    assert id0 == "id_0"

    # --- Test last element (index 11) ---
    x11, y11, w11, id11 = dummy_disk_dataset[11]
    np.testing.assert_array_equal(x11, np.array([22., 23.], dtype=np.float32))
    np.testing.assert_array_equal(y11, np.array([11.], dtype=np.float32))
    assert id11 == "id_11"


@pytest.mark.torch
def test_getitem_shard_boundaries(dummy_disk_dataset):
    """
    Tests retrieving elements at the boundaries of shards to ensure correct
    shard and local index calculation.
    """
    # Shard sizes: [0, 5, 3, 4]. Shards are 0-4, 5-7, 8-11.

    # --- Test last element of first shard (index 4) ---
    x4, y4, _, id4 = dummy_disk_dataset[4]
    np.testing.assert_array_equal(x4, np.array([8., 9.], dtype=np.float32))
    assert id4 == "id_4"

    # --- Test first element of second shard (index 5) ---
    x5, y5, _, id5 = dummy_disk_dataset[5]
    np.testing.assert_array_equal(x5, np.array([10., 11.], dtype=np.float32))
    assert id5 == "id_5"

    # --- Test first element of third shard (index 8) ---
    x8, y8, _, id8 = dummy_disk_dataset[8]
    np.testing.assert_array_equal(x8, np.array([16., 17.], dtype=np.float32))
    assert id8 == "id_8"


@pytest.mark.torch
def test_getitem_out_of_bounds(dummy_disk_dataset):
    """Tests that an IndexError is raised for out-of-bounds access."""
    dataset_len = len(dummy_disk_dataset)

    # Test positive out-of-bounds
    with pytest.raises(IndexError):
        _ = dummy_disk_dataset[dataset_len]


@pytest.mark.torch
def test_collate_dataset_wrapper(dummy_disk_dataset):
    """
    Tests the collate_dataset_wrapper function to ensure it processes batches
    correctly through a DeepChem model.
    """
    # Import required modules
    import torch
    from deepchem.models.fcnet import MultitaskClassifier
    from deepchem.molnet import load_clintox

    _, (_, dataset, _), _ = load_clintox()

    # Create a simple model for testing
    model = MultitaskClassifier(n_tasks=1,
                                n_features=2,
                                layer_sizes=[10],
                                device="cpu",
                                batch_size=4)

    # Create a DataLoader that uses collate_dataset_wrapper
    data_loader = DataLoader(
        _TorchIndexDiskDataset(dataset),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_dataset_fn(batch, model))

    # Get a single batch from the DataLoader
    batch_result = next(iter(data_loader))

    # The collate function should return a tuple of (inputs, labels, weights)
    assert isinstance(batch_result, tuple)
    assert len(batch_result) == 3

    inputs, labels, weights = batch_result

    # Each should be a list of torch tensors
    assert isinstance(inputs, list)
    assert isinstance(labels, list)
    assert isinstance(weights, list)

    # Check that we have processed data (non-empty lists)
    assert len(inputs) > 0
    assert len(labels) > 0
    assert len(weights) > 0

    # Check that elements are torch tensors
    for tensor in inputs:
        assert isinstance(tensor, torch.Tensor)
    for tensor in labels:
        assert isinstance(tensor, torch.Tensor)
    for tensor in weights:
        assert isinstance(tensor, torch.Tensor)
