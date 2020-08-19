from typing import List, Union
import numpy as np
import torch

from deepchem.utils.save import load_image_files
from deepchem.data.datasets import NumpyDataset, DiskDataset, ImageDataset


class _TorchNumpyDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self, numpy_dataset: NumpyDataset, epochs: int,
               deterministic: bool):
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
    """
    self.numpy_dataset = numpy_dataset
    self.epochs = epochs
    self.deterministic = deterministic

  def __iter__(self):
    n_samples = self.numpy_dataset._X.shape[0]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      first_sample = 0
      last_sample = n_samples
    else:
      first_sample = worker_info.id * n_samples // worker_info.num_workers
      last_sample = (worker_info.id + 1) * n_samples // worker_info.num_workers
    for epoch in range(self.epochs):
      if self.deterministic:
        order = first_sample + np.arange(last_sample - first_sample)
      else:
        order = first_sample + np.random.permutation(last_sample - first_sample)
      for i in order:
        yield (self.numpy_dataset._X[i], self.numpy_dataset._y[i],
               self.numpy_dataset._w[i], self.numpy_dataset._ids[i])


class _TorchDiskDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self, disk_dataset: DiskDataset, epochs: int,
               deterministic: bool):
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
    """
    self.disk_dataset = disk_dataset
    self.epochs = epochs
    self.deterministic = deterministic

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    n_shards = self.disk_dataset.get_number_shards()
    if worker_info is None:
      first_shard = 0
      last_shard = n_shards
    else:
      first_shard = worker_info.id * n_shards // worker_info.num_workers
      last_shard = (worker_info.id + 1) * n_shards // worker_info.num_workers
    if first_shard == last_shard:
      return

    shard_indices = list(range(first_shard, last_shard))
    for epoch in range(self.epochs):
      for X, y, w, ids in self.disk_dataset._iterbatches_from_shards(
          shard_indices, deterministic=self.deterministic):
        for i in range(X.shape[0]):
          yield (X[i], y[i], w[i], ids[i])


class _TorchImageDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self, image_dataset: ImageDataset, epochs: int,
               deterministic: bool):
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
    """
    self.image_dataset = image_dataset
    self.epochs = epochs
    self.deterministic = deterministic

  def __iter__(self):
    n_samples = self.image_dataset._X_shape[0]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      first_sample = 0
      last_sample = n_samples
    else:
      first_sample = worker_info.id * n_samples // worker_info.num_workers
      last_sample = (worker_info.id + 1) * n_samples // worker_info.num_workers
    for epoch in range(self.epochs):
      if self.deterministic:
        order = first_sample + np.arange(last_sample - first_sample)
      else:
        order = first_sample + np.random.permutation(last_sample - first_sample)
      for i in order:
        yield (self._get_image(self.image_dataset._X, i),
               self._get_image(self.image_dataset._y, i),
               self.image_dataset._w[i], self.image_dataset._ids[i])

  def _get_image(self, array: Union[np.ndarray, List[str]],
                 index: int) -> np.ndarray:
    """Method for loading an image

    Parameters
    ----------
    array: Union[np.ndarray, List[str]]
      A numpy array which contains images or List of image filenames
    index: int
      Index you want to get the image

    Returns
    -------
    np.ndarray
      Loaded image
    """
    if isinstance(array, np.ndarray):
      return array[index]
    return load_image_files([array[index]])[0]
