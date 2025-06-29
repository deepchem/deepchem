from deepchem.models.lightning.new_dc_lightning_dataset_module import DeepChemLightningDataModule
from deepchem.models.lightning.new_dc_lightning_module import DeepChemLightningModule
from deepchem.models.torch_models import TorchModel
from rdkit import rdBase
import lightning as L
from typing import List, Optional
from deepchem.utils.typing import OneOrMany
from deepchem.trans import Transformer
from deepchem.data import Dataset

rdBase.DisableLog('rdApp.warning')


class DeepChemLightningTrainer:
    """A wrapper class that handles the training and inference of DeepChem models using Lightning.

    This class provides a high-level interface for training and running inference
    on DeepChem models using PyTorch Lightning's training infrastructure. It wraps
    DeepChem models in Lightning modules and handles data loading, training loops,
    and checkpoint management.

    Parameters
    ----------
    model: TorchModel
        Initialized DeepChem model to be trained or used for inference.
    batch_size: int, default 32
        Batch size for training and prediction data loaders.
    **trainer_kwargs
        Additional keyword arguments passed to the Lightning Trainer.

    Examples
    --------
    >>> import deepchem as dc
    >>> import lightning as L
    >>> from deepchem.models.lightning.trainer2 import DeepChemLightningTrainer
    >>> tasks, datasets, _ = dc.molnet.load_clintox()
    >>> _, valid_dataset, _ = datasets
    >>> model = dc.models.MultitaskClassifier(
    ...     n_tasks=len(tasks),
    ...     n_features=1024,
    ...     layer_sizes=[1000],
    ...     dropouts=0.2,
    ...     learning_rate=0.0001,
    ...     device="cpu",
    ...     batch_size=16
    ... )
    >>> trainer = DeepChemLightningTrainer(
    ...     model=model,
    ...     batch_size=16,
    ...     max_epochs=30,
    ...     accelerator="cpu",
    ...     log_every_n_steps=1,
    ...     fast_dev_run=True
    ... )
    >>> trainer.fit(valid_dataset)
    >>> predictions = trainer.predict(valid_dataset)
    >>> trainer.save_checkpoint("model.ckpt")
    >>> # To reload:
    >>> trainer2 = DeepChemLightningTrainer.load_checkpoint("model.ckpt", model=model)
    """

    def __init__(self,
                 model: TorchModel,
                 batch_size: int = 32,
                 **trainer_kwargs):
        self.model = model
        self.batch_size = batch_size
        self.trainer_kwargs = trainer_kwargs

        # Set default trainer arguments if not provided
        if 'accelerator' not in trainer_kwargs:
            self.trainer_kwargs['accelerator'] = 'auto'
        if 'devices' not in trainer_kwargs:
            self.trainer_kwargs['devices'] = 'auto'
        if 'strategy' not in trainer_kwargs:
            self.trainer_kwargs['strategy'] = 'auto'
        if 'max_epochs' not in trainer_kwargs:
            self.trainer_kwargs['max_epochs'] = 100

        # TODO: set up logger if not provided

        # Create the Lightning module
        self.lightning_model = DeepChemLightningModule(model)

    def fit(self, train_dataset: Dataset, num_workers: int = 4):
        """Train the model on the provided dataset.

        Parameters
        ----------
        train_dataset: dc.data.Dataset
            DeepChem dataset for training.
        num_workers: int, default 4
            Number of workers for DataLoader.

        Returns
        -------
        None
            The trainer object is modified in place after fitting.
        """
        # Set log_every_n_steps if not provided
        if 'log_every_n_steps' not in self.trainer_kwargs:
            dataset_size = len(train_dataset)
            self.trainer_kwargs['log_every_n_steps'] = max(
                1, dataset_size // (self.batch_size * 2))

        # Create data module
        data_module = DeepChemLightningDataModule(dataset=train_dataset,
                                                  batch_size=self.batch_size,
                                                  num_workers=num_workers,
                                                  model=self.model)

        # Create trainer
        self.trainer = L.Trainer(**self.trainer_kwargs)

        # Train the model
        self.trainer.fit(self.lightning_model, data_module)

    def predict(self,
                dataset: Dataset,
                transformers: List[Transformer] = [],
                other_output_types: Optional[OneOrMany[str]] = None,
                num_workers: int = 4,
                uncertainty: Optional[bool] = None):
        """Run inference on the provided dataset.

        Parameters
        ----------
        dataset: dc.data.Dataset
            DeepChem dataset for prediction.
        transformers: List[Transformer], default []
            List of transformers to apply to predictions.
        other_output_types: Optional[OneOrMany[str]], default None
            List of other output types to compute.
        num_workers: int, default 4
            Number of workers for DataLoader.
        uncertainty: Optional[bool], default None
            Whether to compute uncertainty estimates.

        Returns
        -------
        List
            Predictions from the model.
        """
        if not hasattr(self, 'trainer'):
            self.trainer = L.Trainer(**self.trainer_kwargs)

        # Create data module
        data_module = DeepChemLightningDataModule(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  num_workers=num_workers,
                                                  model=self.model)
        self.lightning_model.eval()
        # Run prediction

        self.lightning_model._transformers = transformers
        self.lightning_model.other_output_types = other_output_types
        if uncertainty is not None:
            self.lightning_model.uncertainty = uncertainty

        predictions = self.trainer.predict(self.lightning_model,
                                           data_module,
                                           return_predictions=True)

        return predictions

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint.

        Parameters
        ----------
        filepath: str
            Path to save checkpoint.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if hasattr(self, 'trainer'):
            self.trainer.save_checkpoint(filepath)
        else:
            raise ValueError(
                "Model has not been trained yet. Please call fit() first.")

    @staticmethod
    def load_checkpoint(filepath: str,
                        model: TorchModel,
                        batch_size: int = 32,
                        **trainer_kwargs):
        """Load model from checkpoint and create a new trainer instance.

        Parameters
        ----------
        filepath: str
            Path to checkpoint.
        model: TorchModel
            DeepChem model instance to load weights into.
        batch_size: int, default 32
            Batch size for the trainer/model.
        **trainer_kwargs
            Additional trainer arguments.

        Returns
        -------
        DeepChemLightningTrainer
            New trainer instance with loaded model.
        """
        # Load the lightning module from checkpoint
        lightning_model = DeepChemLightningModule.load_from_checkpoint(
            filepath, model=model)

        # Create a new trainer instance
        trainer = DeepChemLightningTrainer(model=model,
                                           batch_size=batch_size,
                                           **trainer_kwargs)

        # Replace the lightning model with the loaded one
        trainer.lightning_model = lightning_model

        return trainer
