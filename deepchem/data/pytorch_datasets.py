import numpy as np
import torch
import torch.distributed as dist
from deepchem.data.datasets import NumpyDataset, DiskDataset, ImageDataset
from typing import Optional, List, Tuple
import bisect


class _TorchNumpyDataset(torch.utils.data.IterableDataset):  # type: ignore

    def __init__(self,
                 numpy_dataset: NumpyDataset,
                 epochs: int,
                 deterministic: bool,
                 batch_size: Optional[int] = None):
        """
        Parameters
        ----------
        numpy_dataset: NumpyDataset
            The original NumpyDataset which you want to convert to PyTorch Dataset
        epochs: int
            the number of times to iterate over the Dataset
        deterministic: bool
            if True, the data is produced in order.  If False, a different random
            permutation of the data is used for each epoch.
        batch_size: int
            the number of samples to return in each batch.  If None, each returned
            value is a single sample.
        """
        self.numpy_dataset = numpy_dataset
        self.epochs = epochs
        self.deterministic = deterministic
        self.batch_size = batch_size

    def __iter__(self):
        n_samples = self.numpy_dataset._X.shape[0]
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            first_sample = 0
            last_sample = n_samples
        else:
            first_sample = worker_info.id * n_samples // worker_info.num_workers
            last_sample = (worker_info.id +
                           1) * n_samples // worker_info.num_workers
        for epoch in range(self.epochs):
            if self.deterministic:
                order = first_sample + np.arange(last_sample - first_sample)
            else:
                # Ensure that every worker will pick the same random order for each epoch.
                random = np.random.RandomState(epoch)
                order = random.permutation(n_samples)[first_sample:last_sample]
            if self.batch_size is None:
                for i in order:
                    yield (self.numpy_dataset._X[i], self.numpy_dataset._y[i],
                           self.numpy_dataset._w[i], self.numpy_dataset._ids[i])
            else:
                for i in range(0, len(order), self.batch_size):
                    indices = order[i:i + self.batch_size]
                    yield (self.numpy_dataset._X[indices],
                           self.numpy_dataset._y[indices],
                           self.numpy_dataset._w[indices],
                           self.numpy_dataset._ids[indices])


class _TorchDiskDataset(torch.utils.data.IterableDataset):  # type: ignore

    def __init__(self,
                 disk_dataset: DiskDataset,
                 epochs: int,
                 deterministic: bool,
                 batch_size: Optional[int] = None):
        """
        Parameters
        ----------
        disk_dataset: DiskDataset
            The original DiskDataset which you want to convert to PyTorch Dataset
        epochs: int
            the number of times to iterate over the Dataset
        deterministic: bool
            if True, the data is produced in order.  If False, a different random
            permutation of the data is used for each epoch.
        batch_size: int
            the number of samples to return in each batch.  If None, each returned
            value is a single sample.
        """
        self.disk_dataset = disk_dataset
        self.epochs = epochs
        self.deterministic = deterministic
        self.batch_size = batch_size

    def __len__(self):
        return len(self.disk_dataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        n_shards = self.disk_dataset.get_number_shards()
        if worker_info is None:
            process_id = 0
            num_processes = 1
        else:
            process_id = worker_info.id
            num_processes = worker_info.num_workers

        if dist.is_initialized():
            process_id += dist.get_rank() * num_processes
            num_processes *= dist.get_world_size()

        first_shard = (process_id * n_shards) // num_processes
        last_shard = ((process_id + 1) * n_shards) // num_processes
        if first_shard == last_shard:
            return

        shard_indices = list(range(first_shard, last_shard))
        for X, y, w, ids in self.disk_dataset._iterbatches_from_shards(
                shard_indices,
                batch_size=self.batch_size,
                epochs=self.epochs,
                deterministic=self.deterministic):
            if self.batch_size is None:
                for i in range(X.shape[0]):
                    yield (X[i], y[i], w[i], ids[i])
            else:
                yield (X, y, w, ids)


class _TorchImageDataset(torch.utils.data.IterableDataset):  # type: ignore

    def __init__(self,
                 image_dataset: ImageDataset,
                 epochs: int,
                 deterministic: bool,
                 batch_size: Optional[int] = None):
        """
        Parameters
        ----------
        image_dataset: ImageDataset
            The original ImageDataset which you want to convert to PyTorch Dataset
        epochs: int
            the number of times to iterate over the Dataset
        deterministic: bool
            if True, the data is produced in order.  If False, a different random
            permutation of the data is used for each epoch.
        batch_size: int
            the number of samples to return in each batch.  If None, each returned
            value is a single sample.
        """
        self.image_dataset = image_dataset
        self.epochs = epochs
        self.deterministic = deterministic
        self.batch_size = batch_size

    def __iter__(self):
        n_samples = self.image_dataset._X_shape[0]
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            first_sample = 0
            last_sample = n_samples
        else:
            first_sample = worker_info.id * n_samples // worker_info.num_workers
            last_sample = (worker_info.id +
                           1) * n_samples // worker_info.num_workers
        for epoch in range(self.epochs):
            if self.deterministic:
                order = first_sample + np.arange(last_sample - first_sample)
            else:
                # Ensure that every worker will pick the same random order for each epoch.
                random = np.random.RandomState(epoch)
                order = random.permutation(n_samples)[first_sample:last_sample]
            if self.batch_size is None:
                for i in order:
                    yield (self.image_dataset._get_image(
                        self.image_dataset._X, i),
                           self.image_dataset._get_image(
                               self.image_dataset._y, i),
                           self.image_dataset._w[i], self.image_dataset._ids[i])
            else:
                for i in range(0, len(order), self.batch_size):
                    indices = order[i:i + self.batch_size]
                    yield (self.image_dataset._get_image(
                        self.image_dataset._X, indices),
                           self.image_dataset._get_image(
                               self.image_dataset._y,
                               indices), self.image_dataset._w[indices],
                           self.image_dataset._ids[indices])


class _TorchIndexDiskDataset(torch.utils.data.Dataset):
    """A wrapper for diskdataset that returns the index of each item in the dataset.

    This wrapper provides random-access indexing for DeepChem datasets,
    making them compatible with PyTorch's DataLoader and enabling efficient
    distributed training scenarios. It implements the map-style dataset
    interface by adding `__getitem__` functionality to datasets that
    typically only support iterator-based access.

    Examples
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from torch.utils.data import DataLoader
    >>> from deepchem.data import _TorchIndexDiskDataset as TorchIndexDiskDataset
    >>>
    >>> # Create a DiskDataset from numpy arrays
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.rand(100, 1)
    >>> w = np.ones((100, 1))
    >>> ids = np.arange(100)
    >>> dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    >>>
    >>> # Wrap the DiskDataset for random access
    >>> wrapped_dataset = TorchIndexDiskDataset(dataset)
    >>>
    >>> # Access individual samples by index
    >>> x_sample, y_sample, w_sample, id_sample = wrapped_dataset[0]
    >>>
    >>> # Use with PyTorch DataLoader
    >>> dataloader = DataLoader(wrapped_dataset, batch_size=16, shuffle=True)
    >>> for batch in dataloader:
    ...     X_batch, y_batch, w_batch, ids_batch = batch
    ...     # Process batch data
    ...     break
    >>>
    """

    def __init__(self, dataset: "DiskDataset"):
        """Initialize the wrapper with a DeepChem dataset.

        Parameters
        ----------
        dataset: DiskDataset
            The DeepChem DiskDataset to wrap for random access.
        """
        self.dataset = dataset
        self._have_cumulative_sums: bool = False
        self._cumulative_sums: List[int] = []

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns
        -------
        int
            The total number of samples across all shards in the dataset.
        """
        return len(self.dataset)

    def _cumulative_sum(self) -> List[int]:
        """Calculate cumulative shard sizes for efficient index mapping.

        This method iterates through all shards once to determine their sizes
        (number of samples) and computes the cumulative sum. The resulting list
        is cached and used by `__getitem__` to quickly map a global sample index
        to a specific shard and a local index within that shard.

        For example, if shard sizes are `[1000, 1000, 500]`, this method
        will return `[0, 1000, 2000, 2500]`. This allows `__getitem__` to use a
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
            self.dataset._get_shard_shape(i)[0][0]  # type: ignore[attr-defined]
            for i in range(
                self.dataset.get_number_shards())  # type: ignore[attr-defined]
        ]
        current_sum = 0
        cumulative_sums = [0] + [
            (current_sum := current_sum + size)  # noqa: F841
            for size in self._shard_sizes
        ]
        return cumulative_sums

    def __getitem__(
        self, index: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray]]:
        """Enable random-access lookup of a single sample by its index.

        This method allows the dataset to be indexed like a standard Python
        list (e.g., `dataset[i]`), returning the i-th sample from across all
        shards. It uses a pre-computed list of cumulative shard sizes to
        efficiently locate the correct shard on disk, loads that shard into
        memory, and returns the specific sample.

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
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            A tuple `(X, y, w, ids)` containing the data for the requested sample.
            Each element can be None if the corresponding data is not available.
        """
        if not self._have_cumulative_sums:
            self._have_cumulative_sums = True
            self._cumulative_sums = self._cumulative_sum()

        # Determine which shard contains the index
        shard_index = bisect.bisect_right(self._cumulative_sums, index) - 1
        # Calculate local index within the shard
        local_index = index - self._cumulative_sums[shard_index]

        # Load the shard data
        shard = self.dataset.get_shard(  # type: ignore[attr-defined]
            shard_index)
        X, y, w, ids = shard

        # Extract the sample (assuming X and y are present)
        X_sample = X[local_index] if X is not None else None
        y_sample = y[local_index] if y is not None else None
        w_sample = w[local_index] if w is not None else None
        ids_sample = ids[local_index] if ids is not None else None

        return (X_sample, y_sample, w_sample, ids_sample)
