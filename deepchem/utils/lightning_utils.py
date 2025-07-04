from torch.utils.data import Dataset as TorchDataset
from typing import List, Tuple, Any, Optional, TYPE_CHECKING
import bisect
import numpy as np
import torch
import deepchem as dc

if TYPE_CHECKING:
    from deepchem.models.torch_models import TorchModel
    from deepchem.data.datasets import DiskDataset


def collate_dataset_wrapper(
    batch_data: List[Tuple[Any, Any, Any, Any]], model: "TorchModel"
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Collate function for DeepChem datasets to work with PyTorch DataLoader.

    This function takes a batch of data from a PyTorch DataLoader and converts
    it into a format compatible with DeepChem models by wrapping it in a
    DeepChemBatch class that processes the data through the model's default
    generator and batch preparation methods.

    It does 3 important operations:
    1. Extracts the features (X), labels (Y), weights (W), and ids from the batch and arranges them correctly.
    2. Creates a NumpyDataset from these components and passes it to the model's default generator.
    3. Calls the model's `_prepare_batch` method that outputs the processed batch as a tuple of tensors.

    Parameters
    ----------
    batch: List[Tuple[Any, Any, Any, Any]]
        Batch of data from DataLoader containing tuples of (X, y, w, ids).
    model: TorchModel
        DeepChem model instance used for batch processing.

    Returns
    -------
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
        Processed batch tuple prepared by the model's _prepare_batch method.

    Examples
    --------
    >>> import deepchem as dc
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from deepchem.models.lightning.utils import collate_dataset_wrapper, IndexDiskDatasetWrapper
    >>>
    >>> # Load a dataset and create a model
    >>> tasks, datasets, _ = dc.molnet.load_clintox()
    >>> _, valid_dataset, _ = datasets
    >>> model = dc.models.MultitaskClassifier(
    ...     n_tasks=len(tasks),
    ...     n_features=1024,
    ...     layer_sizes=[1000],
    ...     device="cpu",
    ...     batch_size=16
    ... )
    >>>
    >>> # Create DataLoader with custom collate function
    >>> wrapped_dataset = IndexDiskDatasetWrapper(valid_dataset)
    >>> dataloader = DataLoader(
    ...     wrapped_dataset,
    ...     batch_size=16,
    ...     collate_fn=lambda batch: collate_dataset_wrapper(batch, model)
    ... )
    >>>
    >>> # Use in training loop
    >>> for batch in dataloader:
    ...     inputs, labels, weights = batch
    ...     # inputs, labels, weights are now properly formatted torch tensors
    ...     break
    """

    X, Y, W, ids = [], [], [], []
    for i in range(len(batch_data)):
        X.append(batch_data[i][0])
        Y.append(batch_data[i][1])
        W.append(batch_data[i][2])
        ids.append(batch_data[i][3])
    processed_batch = next(
        iter(model.default_generator(dc.data.NumpyDataset(X, Y, W, ids))))
    return model._prepare_batch(processed_batch)


class IndexDiskDatasetWrapper(TorchDataset):
    """A wrapper for diskdataset that returns the index of each item in the dataset.

    This wrapper provides random-access indexing for DeepChem datasets,
    making them compatible with PyTorch's DataLoader and enabling efficient
    distributed training scenarios. It implements the map-style dataset
    interface by adding `__getitem__` functionality to datasets that
    typically only support iterator-based access.

    Parameters
    ----------
    dataset: DiskDataset
        The DeepChem DiskDataset to wrap.

    Examples
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from torch.utils.data import DataLoader
    >>> from deepchem.models.lightning.utils import IndexDiskDatasetWrapper
    >>>
    >>> # Create a DiskDataset from numpy arrays
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.rand(100, 1)
    >>> w = np.ones((100, 1))
    >>> ids = np.arange(100)
    >>> dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
    >>>
    >>> # Wrap the DiskDataset for random access
    >>> wrapped_dataset = IndexDiskDatasetWrapper(dataset)
    >>>
    >>> # Access individual samples by index
    >>> x_sample, y_sample, w_sample, id_sample = wrapped_dataset[0]
    >>> print(f"Sample 0 shape: X={x_sample.shape}, y={y_sample.shape}")
    >>>
    >>> # Use with PyTorch DataLoader
    >>> dataloader = DataLoader(wrapped_dataset, batch_size=16, shuffle=True)
    >>> for batch in dataloader:
    ...     X_batch, y_batch, w_batch, ids_batch = batch
    ...     # Process batch data
    ...     break
    >>>
    >>> # Check dataset length
    >>> print(f"Dataset length: {len(wrapped_dataset)}")
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
