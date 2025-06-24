from torch.utils.data import Dataset
import deepchem as dc
from deepchem.data.datasets import NumpyDataset
from typing import List
import bisect


def collate_dataset_wrapper(batch, model):
    """
    Collate function for DeepChem datasets to work with PyTorch DataLoader.

    Args:
        batch: Batch of data from DataLoader
        model: DeepChem model instance

    Returns:
        Tuple of (inputs, labels, weights)
    """

    class DeepChemBatch:

        def __init__(self, batch, model):
            X, Y, W, ids = [], [], [], []
            for i in range(len(batch)):
                X.append(batch[i][0])
                Y.append(batch[i][1])
                W.append(batch[i][2])
                ids.append(batch[i][3])
            batch = next(model.default_generator(NumpyDataset(X, Y, W, ids)))
            self.batch_list = model._prepare_batch(batch)

    return DeepChemBatch(batch, model).batch_list


class IndexDiskDatasetWrapper(Dataset):
    """A wrapper for a dataset that returns the index of each item in the dataset.

    This is useful for debugging and logging purposes.
    """

    def __init__(self, dataset: dc.data.Dataset):
        self.dataset = dataset
        self._have_cumulative_sums: bool = False
        self._cumulative_sums: List[int] = []

    def __len__(self):
        return len(self.dataset)

    def _cumulative_sum(self) -> List[int]:
        """Internal helper method to calculate cumulative shard sizes.

        This method iterates through all shards once to determine their sizes
        (number of samples) and computes the cumulative sum. The resulting list
        is cached and used by `__getitem__` to quickly map a global sample index
        to a specific shard and a local index within that shard.

        For example, if shard sizes are `[1000, 1000, 500]`, this method
        will return `[1000, 2000, 2500]`. This allows `__getitem__` to use a
        fast binary search (O(log N) where N is the number of shards) to
        locate the correct shard for any given index, which is significantly
        more efficient than a linear scan.

        Returns
        -------
        List[int]
            A list where element `i` is the sum of the number of samples in
            shards 0 through `i`.
        """
        self._shard_sizes = [
            self.dataset._get_shard_shape(i)[0][0]
            for i in range(self.dataset.get_number_shards())
        ]
        current_sum = 0
        cumulative_sums = [0] + [
            (current_sum := current_sum + size)  # noqa: F841
            for size in self._shard_sizes
        ]
        return cumulative_sums

    def __getitem__(self, index: int):
        """Enables random-access lookup of a single sample by its index.

        This method allows the dataset to be indexed like a standard Python
        list (e.g., `dataset[i]`), returning the i-th sample from across all
        shards. It uses a pre-computed list of cumulative shard sizes to
        efficiently locate the correct shard on disk, loads that shard into
        memory, and returns the specific sample.

        Rationale
        ---------
        This method provides compatibility with PyTorch's `torch.utils.data.Dataset`
        and `DataLoader`. Standard PyTorch `DataLoader` objects are optimized
        for map-style datasets that implement `__len__` and `__getitem__`.

        While `DiskDataset`'s native iterator-based methods (like `iterbatches`)
        are highly efficient for sequential access, they can cause instability
        or deadlocks in complex multi-process data loading scenarios, such as
        multi-GPU training with PyTorch Lightning's FSDP strategy. By
        implementing `__getitem__`, `DiskDataset` can be wrapped directly by
        a standard `DataLoader`, providing a more robust and stable data
        pipeline for distributed training environments.

        Parameters
        ----------
        index: int
            The global index of the sample to retrieve.

        Returns
        -------
        Tuple
            A tuple `(X, y, w, ids)` containing the data for the requested sample.
        """
        if not self._have_cumulative_sums:
            self._have_cumulative_sums = True
            self._cumulative_sums = self._cumulative_sum()

        # Determine which shard contains the index
        shard_index = bisect.bisect_right(self._cumulative_sums, index) - 1
        # Calculate local index within the shard
        local_index = index - self._cumulative_sums[shard_index]

        # Load the shard data
        shard = self.dataset.get_shard(shard_index)
        X, y, w, ids = shard

        # Extract the sample (assuming X and y are present)
        X_sample = X[local_index] if X is not None else None
        y_sample = y[local_index] if y is not None else None
        w_sample = w[local_index] if y is not None else None
        ids_sample = ids[local_index] if ids is not None else None

        return (X_sample, y_sample, w_sample, ids_sample)
