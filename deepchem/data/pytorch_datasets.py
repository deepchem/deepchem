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
      for X, y, w, ids in self._iterbatches_from_shards(
          shard_indices, deterministic=self.deterministic):
        for i in range(X.shape[0]):
          yield (X[i], y[i], w[i], ids[i])

  def _iterbatches_from_shards(self,
                               shard_indices,
                               batch_size=None,
                               epochs=1,
                               deterministic=False,
                               pad_batches=False):
    """Get an object that iterates over batches from a restricted set of shards."""

    def iterate(dataset, batch_size, epochs):
      num_shards = len(shard_indices)
      if deterministic:
        shard_perm = np.arange(num_shards)

      # (ytz): Depending on the application, thread-based pools may be faster
      # than process based pools, since process based pools need to pickle/serialize
      # objects as an extra overhead. Also, as hideously as un-thread safe this looks,
      # we're actually protected by the GIL.
      pool = multiprocessing.dummy.Pool(
          1)  # mp.dummy aliases ThreadPool to Pool

      if batch_size is None:
        num_global_batches = num_shards
      else:
        num_global_batches = math.ceil(dataset.get_shape()[0][0] / batch_size)

      for epoch in range(epochs):
        if not deterministic:
          shard_perm = np.random.permutation(num_shards)
        next_shard = pool.apply_async(dataset.get_shard,
                                      (shard_indices[shard_perm[0]],))
        cur_global_batch = 0
        cur_shard = 0
        carry = None

        while cur_global_batch < num_global_batches:

          X, y, w, ids = next_shard.get()
          if cur_shard < num_shards - 1:
            next_shard = pool.apply_async(
                dataset.get_shard, (shard_indices[shard_perm[cur_shard + 1]],))
          elif epoch == epochs - 1:
            pool.close()

          if carry is not None:
            X = np.concatenate([carry[0], X], axis=0)
            if y is not None:
              y = np.concatenate([carry[1], y], axis=0)
            if w is not None:
              w = np.concatenate([carry[2], w], axis=0)
            ids = np.concatenate([carry[3], ids], axis=0)
            carry = None

          n_shard_samples = X.shape[0]
          cur_local_batch = 0
          if batch_size is None:
            shard_batch_size = n_shard_samples
          else:
            shard_batch_size = batch_size

          if n_shard_samples == 0:
            cur_shard += 1
            if batch_size is None:
              cur_global_batch += 1
            continue

          num_local_batches = math.ceil(n_shard_samples / shard_batch_size)
          if not deterministic:
            sample_perm = np.random.permutation(n_shard_samples)
          else:
            sample_perm = np.arange(n_shard_samples)

          while cur_local_batch < num_local_batches:
            start = cur_local_batch * shard_batch_size
            end = min(n_shard_samples, (cur_local_batch + 1) * shard_batch_size)

            indices = range(start, end)
            perm_indices = sample_perm[indices]
            X_b = X[perm_indices]

            if y is not None:
              y_b = y[perm_indices]
            else:
              y_b = None

            if w is not None:
              w_b = w[perm_indices]
            else:
              w_b = None

            ids_b = ids[perm_indices]

            assert len(X_b) <= shard_batch_size
            if len(X_b) < shard_batch_size and cur_shard != num_shards - 1:
              assert carry is None
              carry = [X_b, y_b, w_b, ids_b]
            else:

              # (ytz): this skips everything except possibly the last shard
              if pad_batches:
                (X_b, y_b, w_b, ids_b) = pad_batch(shard_batch_size, X_b, y_b,
                                                   w_b, ids_b)

              yield X_b, y_b, w_b, ids_b
              cur_global_batch += 1
            cur_local_batch += 1
          cur_shard += 1

    return iterate(self.disk_dataset, batch_size, epochs)


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
