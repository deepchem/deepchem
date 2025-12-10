from typing import Callable, Optional
from enum import Enum
import deepchem as dc
import lightning as L
import torch
from deepchem.utils import collate_dataset_fn
from deepchem.models.torch_models import TorchModel
import logging

logger = logging.getLogger(__name__)


class Stage(Enum):
    """Enum representing different stages for dataset setup."""
    FIT = "fit"
    PREDICT = "predict"


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
            A deepchem dataset.
        batch_size: int
            Batch size for the dataloader.
        collate_fn: Callable
            Method to collate instances across batch.
        num_workers: int
            Number of workers to load data
        model: Optional[TorchModel], Defaults to None
            DeepChem model instance required for proper data preprocessing in newer workflow.
            The model provides custom `default_generator()` and `_prepare_batch()`
            methods that are used by the collate function for model-specific data preprocessing.

        Notes
        -----
        The `model` parameter was introduced to support a new workflow that addresses
        stability issues in distributed training scenarios, particularly with PyTorch
        Lightning's FSDP (Fully Sharded Data Parallel) strategy. The implementation
        has two workflows:

        1. **Legacy workflow** (when `model=None`): Uses the original `make_pytorch_dataset()`
           method, which creates iterable datasets. This approach works great for DDP (Distributed Data Parallel Training),
           but can cause deadlocks or instability in complex multi-process data loading
           scenarios with FSDP.

        2. **New workflow** (when `model` is provided): Uses `_TorchIndexDiskDataset` to
           wrap the dataset with random-access indexing capabilities. This provides:

           - **FSDP compatibility**: The map-style dataset interface (`__getitem__` and
             `__len__`) is more robust for distributed training and avoids deadlocks
             that can occur with iterator-based datasets in FSDP scenarios.

           - **Model-specific data preprocessing**: Sometimes TorchModel subclasses often define
             custom `default_generator()` and `_prepare_batch()` methods for specialized
             data preprocessing. The collate function handles all data preprocessing by
             utilizing these model-specific methods, ensuring proper data format and
             transformations before it enters the train/prediction workflows.
        """
        super().__init__()
        self._batch_size = batch_size

        # Refer to the docstring for details on the two workflows.
        if model is None:
            self._dataset = dataset.make_pytorch_dataset(  # type: ignore[has-type]
                batch_size=self._batch_size)

            self.collate_fn = collate_fn
            self.DDP_ONLY_WORKFLOW = True
            logger.warning(
                "Using DDP-only compatible workflow. Please provide the respective deepchem model to use the new workflow, that is compatible with both FSDP and DDP."
            )
        else:
            self._dataset = dc.data._TorchIndexDiskDataset(
                dataset)  # type: ignore[arg-type]

            # Since the model argument is provided, we assume that the user wants to use the FSDP-DDP compatible workflow, and hence replace the default generator-based collate function (collate_dataset_wrapper)
            # with one that uses an indexable collate function (collate_dataset_fn).
            if collate_fn == collate_dataset_wrapper:
                collate_fn = collate_dataset_fn
            self.collate_fn = lambda batch: collate_fn(batch_data=batch,
                                                       model=model)
            self.DDP_ONLY_WORKFLOW = False
        self.num_workers = num_workers

    def setup(self, stage: str):
        """Set up datasets for each stage.

        Parameters
        ----------
        stage: str
            The stage to set up datasets for ('fit' or 'predict').
        """
        if stage == Stage.FIT.value:
            self.train_dataset = self._dataset
        elif stage == Stage.PREDICT.value:
            self.predict_dataset = self._dataset

    def train_dataloader(self):
        """Returns the train dataloader from train dataset.

        Returns
        -------
        DataLoader
            train dataloader to be used with DCLightningModule.
        """
        # In the fsdp-ddp compatible workflow, batching and shuffling is handled by torch.utils.data.DataLoader.
        # In the DDP-only workflow, we set batch_size to None and shuffle to False, since both are handled by the _TorchDiskDataset(deepchem's iterative pytorch datalaoder).
        if self.DDP_ONLY_WORKFLOW:
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
            predict dataloader to be used with DCLightningModule.
        """
        if self.DDP_ONLY_WORKFLOW:
            batch_size = None
        else:
            batch_size = self._batch_size

        dataloader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,  # Critical: never shuffle during prediction
            num_workers=self.num_workers)

        return dataloader
