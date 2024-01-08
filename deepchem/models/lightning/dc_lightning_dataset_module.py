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
    """DeepChem Lightning Dataset Module to be used with the DCLightningModule and a Lightning trainer.

    This module wraps over the the deepchem pytorch dataset and dataloader providing a generic interface to run training.
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html

    Notes
    -----
    This class requires PyTorch to be installed.
    """

    def __init__(self, dataset, batch_size, collate_fn):
        """Create a new DCLightningDatasetModule.

        Parameters
        ----------
        dataset: A deepchem dataset.
        batch_size: Batch size for the dataloader.
        collate_fn: Method to collate instances across batch.
        """
        super().__init__()
        self._batch_size = batch_size
        self._dataset = dataset.make_pytorch_dataset(
            batch_size=self._batch_size)
        self.collate_fn = collate_fn

    def setup(self, stage):
        self.train_dataset = self._dataset

    def train_dataloader(self):
        """Returns the train dataloader from train dataset.

        Returns
        -------
        dataloader: train dataloader to be used with DCLightningModule.
        """
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=None,
            collate_fn=self.collate_fn,
            shuffle=False,
        )
        return dataloader
