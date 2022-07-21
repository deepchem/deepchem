import pytorch_lightning as pl
import torch


class DCLightningDatasetBatch:

  def __init__(self, batch):
    X = [batch[0]]
    y = [batch[1]]
    w = [batch[2]]
    self.batch_list = [X, y, w]


def collate_dataset_wrapper(batch):
  return DCLightningDatasetBatch(batch)


class DCLightningDatasetModule(pl.LightningDataModule):

  def __init__(self, dataset, batch_size, collate_fn):
    super().__init__()
    self._batch_size = batch_size
    self._dataset = dataset.make_pytorch_dataset(batch_size=self._batch_size)
    self.collate_fn = collate_fn

  def setup(self, stage):
    self.train_dataset = self._dataset

  def train_dataloader(self):
    return torch.utils.data.DataLoader(
        self.train_dataset,
        batch_size=None,
        collate_fn=self.collate_fn,
        shuffle=False,
    )
