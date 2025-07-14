from torch.utils.data import DataLoader
import lightning as L
import deepchem as dc
from deepchem.data import Dataset, TorchIndexDiskDataset
from typing import Optional, Callable
from deepchem.utils import collate_dataset_fn
from deepchem.models.torch_models import TorchModel


class DeepChemLightningDataModule(L.LightningDataModule):
    """A PyTorch Lightning DataModule for DeepChem datasets.

    This DataModule integrates DeepChem dataset handling with PyTorch Lightning's streamlined
    data loading and preprocessing pipeline. It wraps a DeepChem dataset (like `TorchIndexDiskDataset`)
    and configures DataLoaders for training (fit) and prediction (inference) stages. The design supports
    a customizable collate function to ensure that data is properly formatted for consumption by DeepChem
    models. If a DeepChem model is provided and no collate function is specified, it defaults to using a
    collate function (e.g. `collate_dataset_fn`) associated with the model, enabling specialized
    processing of inputs.

    Parameters
    ----------
    dataset: dc.data.Dataset
        DeepChem dataset for training.
    batch_size: int
        Batch size for training.
    collate_fn: Optional[Callable], default None
        Custom collate function. If None and model is provided, defaults to collate_dataset_fn.
    num_workers: int, default 0
        Number of workers for DataLoader.
    model: Optional[TorchModel], default None
        DeepChem model for collate function.

    Notes
    ----
    For more information, see:
      - PyTorch Lightning DataModule Documentation: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 collate_fn: Optional[Callable] = None,
                 num_workers: int = 0,
                 model: Optional[TorchModel] = None):
        """Lightning DataModule for DeepChem datasets.

        Parameters
        ----------
        dataset: dc.data.Dataset
            DeepChem dataset for training.
        batch_size: int
            Batch size for training.
        collate_fn: Optional[Callable], default None
            Custom collate function. If None, defaults to collate_dataset_fn.
        num_workers: int, default 0
            Number of workers for DataLoader.
        model: Optional[TorchModel], default None
            DeepChem model for collate function.
        """
        super().__init__()
        self._batch_size = batch_size
        self._dataset = TorchIndexDiskDataset(dataset)
        self._model = model

        if collate_fn is None and model is not None:
            self.collate_fn = lambda batch: collate_dataset_fn(batch_data=batch,
                                                               model=model)
        else:
            self.collate_fn = collate_fn

        self.num_workers = num_workers

    def setup(self, stage: str):
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
        """Return the training dataloader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Return the prediction dataloader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for prediction data.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,  # Critical: never shuffle during prediction
            num_workers=self.num_workers,
        )
