import unittest
import deepchem as dc
import numpy as np

try:
  from deepchem.models import GCNModel, MultitaskClassifier
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

    class TestMultitaskDatasetBatch:

      def __init__(self, batch):
        X = [np.array([b[0] for b in batch])]
        y = [np.array([b[1] for b in batch])]
        w = [np.array([b[2] for b in batch])]
        self.batch_list = [X, y, w]

    def collate_dataset_wrapper(batch):
      return TestMultitaskDatasetBatch(batch)

    class TestMultitaskDataset(torch.utils.data.Dataset):

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

    valid_dataloader = DataLoader(TestMultitaskDataset(valid_dataset),
                                  batch_size=64,
                                  collate_fn=collate_dataset_wrapper)

    lightning_module = DCLightningModule(model)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(lightning_module, valid_dataloader)

  @unittest.skipIf(PYTORCH_LIGHTNING_IMPORT_FAILED,
                   'PyTorch Lightning is not installed')
  def test_gcn_model(self):

    class TestGCNDataset(torch.utils.data.Dataset):

      def __init__(self, smiles, labels):
        assert len(smiles) == len(labels)
        featurizer = dc.feat.MolGraphConvFeaturizer()
        X = featurizer.featurize(smiles)
        self._samples = dc.data.NumpyDataset(X=X, y=labels)

      def __len__(self):
        return len(self._samples)

      def __getitem__(self, index):
        return (
            self._samples.X[index],
            self._samples.y[index],
            self._samples.w[index],
        )

    class TestGCNDatasetBatch:

      def __init__(self, batch):
        X = [np.array([b[0] for b in batch])]
        y = [np.array([b[1] for b in batch])]
        w = [np.array([b[2] for b in batch])]
        self.batch_list = [X, y, w]

    def collate_gcn_dataset_wrapper(batch):
      return TestGCNDatasetBatch(batch)

    train_smiles = [
        "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC",
        "C1CCC1", "CCC"
    ]
    train_labels = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]

    model = GCNModel(mode='classification',
                     n_tasks=1,
                     batch_size=2,
                     learning_rate=0.001)
    train_dataloader = DataLoader(TestGCNDataset(train_smiles, train_labels),
                                  batch_size=100,
                                  collate_fn=collate_gcn_dataset_wrapper)

    lightning_module = DCLightningModule(model)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(lightning_module, train_dataloader)
