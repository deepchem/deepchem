from typing import Callable, Optional
import deepchem as dc
import lightning as L
import torch
from deepchem.utils import collate_dataset_fn
from deepchem.models.torch_models import TorchModel


class DCLightningDatasetModule(L.LightningDataModule):
    """DeepChem Lightning Dataset Module to be used with the DCLightningModule and a Lightning trainer.

    This module wraps over the the deepchem pytorch dataset and dataloader providing a generic interface to run training.

    Notes
    -----
    This class requires PyTorch and lightning to be installed.
    For more information, see:
      - PyTorch Lightning DataModule Documentation: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self,
                 dataset: dc.data.Dataset,
                 batch_size: int,
                 collate_fn: Optional[Callable] = None,
                 num_workers: int = 0,
                 model: Optional[TorchModel] = None):
        """Create a new DCLightningDatasetModule.

        Parameters
        ----------
        dataset: dc.data.Dataset
            A deepchem dataset.
        batch_size: int
            Batch size for the dataloader.
        collate_fn: Optional[Callable], default None
            Custom collate function. If None and model is provided, defaults to collate_dataset_fn.
        num_workers: int
            Number of workers to load data
        model: Optional[TorchModel], default None
            DeepChem model for collate function.
        """
        super().__init__()
        self._batch_size = batch_size
        self._dataset = dc.data._TorchIndexDiskDataset(dataset)
        if collate_fn is None and model is not None:
            self.collate_fn = lambda batch: collate_dataset_fn(batch_data=batch,
                                                               model=model)
        else:
            self.collate_fn = collate_fn
        self.num_workers = num_workers

    def setup(self, stage):
        """Set up datasets for each stage.

        Parameters
        ----------
        stage: str
            The stage to set up datasets for ('fit' or 'predict').
        """
        if stage == "fit":
            self.train_dataset = self._dataset
        elif stage == "predict":
            self.predict_dataset = self._dataset

    def train_dataloader(self):
        """Returns the train dataloader from train dataset.

        Returns
        -------
        DataLoader
            Pytorch DataLoader for train data.
        """
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def predict_dataloader(self):
        """Return the prediction dataloader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for prediction data.
        """
        dataloader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,  # Critical: never shuffle during prediction
            num_workers=self.num_workers,
            drop_last=True)

        return dataloader
