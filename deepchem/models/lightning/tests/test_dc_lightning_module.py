import unittest
import deepchem as dc
import numpy as np

try:
  from deepchem.models import MultitaskClassifier
  import torch
  from torch.utils.data import DataLoader
  from deepchem.models.lightning.dc_lightning_module import DCLightningModule
  import pytorch_lightning as pl  # noqa
  PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
  PYTORCH_LIGHTNING_IMPORT_FAILED = True


class TestDCLightningModule(unittest.TestCase):

  @unittest.skipIf(PYTORCH_LIGHTNING_IMPORT_FAILED,
                   'PyTorch Lightning is not installed')
  def test_multitask_classifier(self):

    class TestDatasetBatch:

      def __init__(self, batch):
        X = [np.array([b[0] for b in batch])]
        y = [np.array([b[1] for b in batch])]
        w = [np.array([b[2] for b in batch])]
        self.batch_list = [X, y, w]

    def collate_dataset_wrapper(batch):
      return TestDatasetBatch(batch)

    class TestDataset(torch.utils.data.Dataset):

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

    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets

    model = MultitaskClassifier(n_tasks=len(tasks),
                                n_features=1024,
                                layer_sizes=[1000],
                                dropouts=0.2,
                                learning_rate=0.0001)

    valid_dataloader = DataLoader(TestDataset(valid_dataset),
                                  batch_size=64,
                                  collate_fn=collate_dataset_wrapper)

    lightning_module = DCLightningModule(model)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(lightning_module, valid_dataloader)
