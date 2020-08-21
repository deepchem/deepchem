import numpy as np
import torch

from deepchem.data.datasets import NumpyDataset, DiskDataset, ImageDataset


class _TorchNumpyDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self,
               numpy_dataset: NumpyDataset,
               epochs: int,
               deterministic: bool,
               batch_size: int = None):
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
      last_sample = (worker_info.id + 1) * n_samples // worker_info.num_workers
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
          yield (self.numpy_dataset._X[indices], self.numpy_dataset._y[indices],
                 self.numpy_dataset._w[indices],
                 self.numpy_dataset._ids[indices])


class _TorchDiskDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self,
               disk_dataset: DiskDataset,
               epochs: int,
               deterministic: bool,
               batch_size: int = None):
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
               batch_size: int = None):
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
      last_sample = (worker_info.id + 1) * n_samples // worker_info.num_workers
    for epoch in range(self.epochs):
      if self.deterministic:
        order = first_sample + np.arange(last_sample - first_sample)
      else:
        # Ensure that every worker will pick the same random order for each epoch.
        random = np.random.RandomState(epoch)
        order = random.permutation(n_samples)[first_sample:last_sample]
      if self.batch_size is None:
        for i in order:
          yield (self.image_dataset._get_image(self.image_dataset._X, i),
                 self.image_dataset._get_image(self.image_dataset._y, i),
                 self.image_dataset._w[i], self.image_dataset._ids[i])
      else:
        for i in range(0, len(order), self.batch_size):
          indices = order[i:i + self.batch_size]
          yield (self.image_dataset._get_image(self.image_dataset._X, indices),
                 self.image_dataset._get_image(self.image_dataset._y, indices),
                 self.image_dataset._w[indices],
                 self.image_dataset._ids[indices])
