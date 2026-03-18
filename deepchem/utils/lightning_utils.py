from typing import List, Tuple, Any
import deepchem as dc


def collate_dataset_fn(batch_data: List[Tuple[Any, Any, Any, Any]], model):
    """Default Collate function for DeepChem datasets to work with PyTorch DataLoader.

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
    >>> from deepchem.data import _TorchIndexDiskDataset as TorchIndexDiskDataset
    >>> from deepchem.utils.lightning_utils import collate_dataset_fn
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
    >>> wrapped_dataset = TorchIndexDiskDataset(valid_dataset)
    >>> dataloader = DataLoader(
    ...     wrapped_dataset,
    ...     batch_size=16,
    ...     collate_fn=lambda batch: collate_dataset_fn(batch, model)
    ... )
    >>>
    >>> # Use in training loop
    >>> for batch in dataloader:
    ...     inputs, labels, weights = batch
    ...     # inputs, labels, weights are now properly formatted torch tensors
    ...     break
    """

    X, Y, W, ids = [], [], [], []
    X = [item[0] for item in batch_data]
    Y = [item[1] for item in batch_data]
    W = [item[2] for item in batch_data]
    ids = [item[3] for item in batch_data]
    processed_batch = next(
        iter(model.default_generator(dc.data.NumpyDataset(X, Y, W, ids))))
    return model._prepare_batch(processed_batch)
