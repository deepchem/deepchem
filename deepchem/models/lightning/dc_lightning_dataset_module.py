from typing import Callable, Optional
import deepchem as dc
import lightning as L
import torch
from deepchem.utils import collate_dataset_fn
from deepchem.models.torch_models import TorchModel
import logging

logger = logging.getLogger(__name__)


class DCLightningDatasetBatch:

    def __init__(self, batch):
        X = [batch[0]]
        y = [batch[1]]
        w = [batch[2]]
        self.batch_list = [X, y, w]


def collate_dataset_wrapper(batch):
    return DCLightningDatasetBatch(batch)


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
                 collate_fn: Callable = collate_dataset_wrapper,
                 num_workers: int = 0,
                 model: Optional[TorchModel] = None):
        """Create a new DCLightningDatasetModule.

        Parameters
        ----------
        dataset: dc.data.Dataset
            A deepchem loaded dataset.
        batch_size: int
            Batch size for the dataloader.
        collate_fn: Callable
            Custom collate function. Defaults to collate_dataset_wrapper.
        num_workers: int
            Number of workers to load data
        model: Optional[TorchModel], Defaults to None
            DeepChem model for collate function, which is used to utilize the model's
            `_prepare_batch` and `default_generator` method for preparing the batch data.
        """
        super().__init__()
        self._batch_size = batch_size
        if model is None:
            self._dataset = dataset.make_pytorch_dataset(  # type: ignore[has-type]
                batch_size=self._batch_size)

            self.collate_fn = collate_fn
            self.DEPRECATED_WORKFLOW = True
            logger.warning(
                "Using deprecated workflow. Please provide the respective deepchem model to use the new workflow, this will be removed in future versions."
            )
        else:
            self._dataset = dc.data._TorchIndexDiskDataset(
                dataset)  # type: ignore[arg-type]
            if collate_fn == collate_dataset_wrapper:
                collate_fn = collate_dataset_fn
            self.collate_fn = lambda batch: collate_fn(batch_data=batch,
                                                       model=model)
            self.DEPRECATED_WORKFLOW = False
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
        # In the newer workflow, batching is handles by DataLoader.
        if self.DEPRECATED_WORKFLOW:
            batch_size = None
            shuffle = False
        else:
            batch_size = self._batch_size
            shuffle = True

        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
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
        if self.DEPRECATED_WORKFLOW:
            batch_size = None
        else:
            batch_size = self._batch_size

        dataloader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,  # Critical: never shuffle during prediction
            num_workers=self.num_workers,
            drop_last=True)

        return dataloader
