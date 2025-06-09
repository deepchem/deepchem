from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
import deepchem as dc
from deepchem.data.datasets import NumpyDataset
import logging
from typing import Optional,Callable


def collate_dataset_wrapper(batch, model):
    """
    Collate function for DeepChem datasets to work with PyTorch DataLoader.
    
    Args:
        batch: Batch of data from DataLoader
        model: DeepChem model instance
        
    Returns:
        Tuple of (inputs, labels, weights)
    """
    class DeepChemBatch:
        def __init__(self, batch, model):
            X, Y, W, ids = [], [], [], [] 
            for i in range(len(batch)):
                X.append(batch[i][0])
                Y.append(batch[i][1])
                W.append(batch[i][2])
                ids.append(batch[i][3])
            size = model.batch_size # This helps to make sure that the `default_generator` does not batch the data.
            model.batch_size = 1
            batch = next(model.default_generator(NumpyDataset(X, Y, W, ids)))
            self.batch_list = model._prepare_batch(batch)
            model.batch_size = size
    return DeepChemBatch(batch, model).batch_list

class DCLightningDataModule(L.LightningDataModule):
    """
    Lightning DataModule for DeepChem datasets.
    
    Args:
        dataset: DeepChem dataset for training
        batch_size: Batch size for training
        collate_fn: Custom collate function (default: collate_dataset_wrapper)
        num_workers: Number of workers for DataLoader
        model: DeepChem model for collate function
    """
    def __init__(
        self,
        dataset: dc.data.Dataset,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 0,
        model = None
    ):
        super().__init__()
        self._batch_size = batch_size
        self._dataset = dataset
        self._model = model
        
        if collate_fn is None and model is not None:
            self.collate_fn = lambda batch: collate_dataset_wrapper(batch, model)
        else:
            self.collate_fn = collate_fn
            
        self.num_workers = num_workers

    def setup(self, stage: str):
        """Set up datasets for each stage."""
        if stage == "fit":
            self.train_dataset = self._dataset
        elif stage == "predict":
            self.predict_dataset = self._dataset

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Return the prediction dataloader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self._batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
        )