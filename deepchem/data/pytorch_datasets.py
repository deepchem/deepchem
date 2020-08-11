import math
import multiprocessing

import numpy as np
import torch

from deepchem.data.datasets import pad_batch
from deepchem.data.data_loader import ImageLoader


class TorchNumpyDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self, X, y, w, ids, n_samples, epochs, deterministic):
    self._X = X
    self._y = y
    self._w = w
    self._ids = ids
    self.n_samples = n_samples
    self.epochs = epochs
    self.deterministic = deterministic

  def __iter__(self):
    n_samples = self.n_samples
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
        yield (self._X[i], self._y[i], self._w[i], self._ids[i])


class TorchDiskDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self, disk_dataset, epochs, deterministic):
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


class TorchImageDataset(torch.utils.data.IterableDataset):  # type: ignore

  def __init__(self, X, y, w, ids, n_samples, epochs, deterministic):
    self._X = X
    self._y = y
    self._w = w
    self._ids = ids
    self.n_samples = n_samples
    self.epochs = epochs
    self.deterministic = deterministic

  def __iter__(self):
    n_samples = self.n_samples
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
        yield (self._get_image(self._X, i), self._get_image(self._y, i),
               self._w[i], self._ids[i])

  def _get_image(self, array, index):
    if isinstance(array, np.ndarray):
      return array[index]
    return ImageLoader.load_img([array[index]])[0]
