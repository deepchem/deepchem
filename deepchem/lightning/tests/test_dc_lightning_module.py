import unittest
import deepchem as dc
from deepchem.models import MultitaskClassifier
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from deepchem.lightning.dc_lightning_module import DCLightningModule


class MolnetDataset(torch.utils.data.Dataset):

  def __init__(self, dataset):
    self._samples = dataset

  def __len__(self):
    return len(self._samples)

  def __getitem__(self, index):
    y = np.zeros((1, 2))
    y[0, int(self._samples.y[index][0])] = 1.0
    return (
        self._samples.X[index],
        y,
        self._samples.w[index],
    )


class MolnetDatasetBatch:

  def __init__(self, batch):
    X = [np.array([b[0] for b in batch])]
    y = [np.array([b[1] for b in batch])]
    w = [np.array([b[2] for b in batch])]
    self.batch_list = [X, y, w]


def collate_dataset_wrapper(batch):
  return MolnetDatasetBatch(batch)


class TestDCLightningModule(unittest.TestCase):

  def test_multitask_classifier(self):
    tasks, datasets, _ = dc.molnet.load_hiv(featurizer='ECFP',
                                            splitter='scaffold')
    train_dataset, _, _ = datasets

    model = MultitaskClassifier(n_tasks=len(tasks),
                                n_features=1024,
                                layer_sizes=[1000],
                                dropouts=0.2,
                                learning_rate=0.0001)

    train_dataloader = DataLoader(MolnetDataset(train_dataset),
                                  batch_size=64,
                                  collate_fn=collate_dataset_wrapper)

    lightning_module = DCLightningModule(model)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(lightning_module, train_dataloader)
