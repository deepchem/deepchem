from deepchem.data import Dataset
from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule
from deepchem.models.lightning.dc_lightning_module import DCLightningModule
from deepchem.models.torch_models import TorchModel
from deepchem.trans import Transformer
from deepchem.utils.typing import OneOrMany
from typing import Any, Dict, List, Optional
from deepchem.models import Model
import logging
import numpy as np
import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')
logger = logging.getLogger(__name__)


class LightningTorchModel(Model):
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
    model_dir: str, (default "default_model_dir")
        Path to directory where model and checkpoints will be stored. If not specified,
        model will be stored in a "default_model_dir" directory. This is compatible with
        DeepChem's model directory structure. If given as None, a temporary directory will be used.
    **trainer_kwargs
        Additional keyword arguments passed to the Lightning Trainer. Common options include:

            - max_epochs: int, default None
                Maximum number of training epochs.
            - accelerator: str, default "auto"
                Hardware accelerator to use ("cpu", "gpu", "tpu", "auto").
            - devices: int or str or list, default "auto"
                Number of devices/GPUs to use.
            - strategy: str, default "auto"
                Distributed training strategy ("ddp", "fsdp", "auto").
            - precision: str or int, default "32-true"
                Numerical precision ("16-mixed", "bf16-mixed", "32-true").
            - log_every_n_steps: int, default 50
                How often to log within training steps.
            - enable_checkpointing: bool, default True
                Whether to enable automatic checkpointing.
            - fast_dev_run: bool or int, default False
                Run a fast development run with limited batches for debugging.
        For all available options, see: https://lightning.ai/docs/pytorch/stable/common/trainer.html#init


    Examples
    --------
    >>> import deepchem as dc
    >>> import lightning as L
    >>> from deepchem.models.lightning.trainer import LightningTorchModel
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
    >>> trainer = LightningTorchModel(
    ...     model=model,
    ...     batch_size=16,
    ...     max_epochs=30,
    ...     accelerator="cpu",
    ...     log_every_n_steps=1,
    ...     fast_dev_run=True
    ... )
    >>> # Train with custom checkpoint settings
    >>> trainer.fit(valid_dataset,
    ...              max_checkpoints_to_keep=3,
    ...              checkpoint_interval=1000)
    >>> predictions = trainer.predict(valid_dataset)
    >>> trainer.save_checkpoint("model.ckpt")
    >>> # To reload:
    >>> trainer2 = LightningTorchModel.load_checkpoint("model.ckpt", model=model)
    """

    def __init__(self,
                 model: TorchModel,
                 batch_size: int = 32,
                 model_dir: Optional[str] = "default_model_dir",
                 **trainer_kwargs: Any) -> None:

        self.model: TorchModel = model
        self.batch_size: int = batch_size
        self.trainer_kwargs: Dict[str, Any] = trainer_kwargs

        assert model.batch_size == batch_size, \
            "Model's batch size must match the LightningTorchModel's batch size."

        # checkpointing is enabled by default
        if 'enable_checkpointing' not in self.trainer_kwargs:
            self.trainer_kwargs['enable_checkpointing'] = True
        else:
            if not self.trainer_kwargs['enable_checkpointing']:
                model_dir = None  # Disable checkpointing if explicitly set to False

        # Set default_root_dir for Lightning to use our model_dir if not specified
        if 'default_root_dir' not in self.trainer_kwargs:
            self.trainer_kwargs['default_root_dir'] = model_dir

        # Create the Lightning module
        self.lightning_model: DCLightningModule = DCLightningModule(model)

        # Initialize the base Model class with model_dir to ensure compatibility
        super(LightningTorchModel, self).__init__(model=model,
                                                  model_dir=model_dir)

    def fit(self,
            train_dataset: Dataset,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 1000,
            num_workers: int = 4,
            ckpt_path: Optional[str] = None):
        """Train the model on the provided dataset.

        Parameters
        ----------
        train_dataset: dc.data.Dataset
            DeepChem dataset for training.
        max_checkpoints_to_keep: int, default 5
            The maximum number of checkpoints to keep.
        checkpoint_interval: int, default 1000
            The frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        num_workers: int, default 4
            Number of workers for DataLoader.
        ckpt_path: Optional[str], default None
            Path to a checkpoint file to resume training from. If None, starts fresh.
        """

        # Prepare callbacks for the trainer
        callbacks_list = []

        # Add checkpoint callback if interval > 0
        if checkpoint_interval > 0:
            # Use Lightning's built-in checkpoint rotation by monitoring "global_step"
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(self.model_dir, "checkpoints"),
                filename='epoch={epoch}-step={step}',
                monitor='step',  # Monitor the step count
                mode='max',  # Keep checkpoints with highest step values
                every_n_train_steps=checkpoint_interval,
                save_top_k=max_checkpoints_to_keep,
                save_last=True,  # Always keep the last checkpoint
                auto_insert_metric_name=False,
                verbose=True)
            callbacks_list.append(checkpoint_callback)

        # Check if trainer already has callbacks from trainer_kwargs
        if 'callbacks' in self.trainer_kwargs:
            existing_callbacks = self.trainer_kwargs['callbacks'] or []
            # Filter out any existing ModelCheckpoint callbacks and add our new one
            other_callbacks = [
                cb for cb in existing_callbacks
                if not isinstance(cb, ModelCheckpoint)
            ]
            callbacks_list.extend(other_callbacks)

        # Create a new trainer with the updated callbacks if we have checkpoint callback
        if checkpoint_interval > 0:
            # Update trainer_kwargs to include our callbacks
            updated_kwargs = self.trainer_kwargs.copy()
            updated_kwargs['callbacks'] = callbacks_list
            self.trainer = L.Trainer(**updated_kwargs)
        else:
            self.trainer = L.Trainer(**self.trainer_kwargs)

        # Create data module
        data_module = DCLightningDatasetModule(dataset=train_dataset,
                                               batch_size=self.batch_size,
                                               num_workers=num_workers,
                                               model=self.model)

        # Train the model
        self.trainer.fit(self.lightning_model, data_module, ckpt_path=ckpt_path)

    def predict(self,
                dataset: Dataset,
                transformers: List[Transformer] = [],
                other_output_types: Optional[OneOrMany[str]] = None,
                num_workers: int = 0,
                uncertainty: Optional[bool] = None,
                ckpt_path: Optional[str] = None):
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
        ckpt_path: Optional[str], default None
            Path to a checkpoint file to load model weights from.

        Returns
        -------
        List
            Predictions from the model.
        """
        self.trainer = L.Trainer(**self.trainer_kwargs)

        # Create data module
        data_module = DCLightningDatasetModule(dataset=dataset,
                                               batch_size=self.batch_size,
                                               num_workers=num_workers,
                                               model=self.model)

        # Set prediction parameters
        self.lightning_model.transformers = transformers
        self.lightning_model.other_output_types = other_output_types

        if uncertainty is not None:
            self.lightning_model.uncertainty = uncertainty

        # Run prediction
        predictions = self.trainer.predict(self.lightning_model,
                                           datamodule=data_module,
                                           return_predictions=True,
                                           ckpt_path=ckpt_path)

        if predictions:
            try:
                predictions = np.concatenate([p for p in predictions])
            except ValueError:
                # If concatenation fails (e.g., due to different shapes in MLM tasks),
                # return the predictions as a list of batches
                pass
        else:
            predictions = []
        return predictions

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint using Lightning's native checkpointing.

        This method saves a complete checkpoint containing the model state,
        optimizer state, learning rate scheduler state (if any schedulers are
        configured), current training epoch, step counts, and other training
        related metadata.

        Parameters
        ----------
        filepath: str
            Path to save the checkpoint file (.ckpt extension recommended).
        """

        self.trainer.save_checkpoint(filepath)

    @staticmethod
    def load_checkpoint(filepath: str,
                        model: TorchModel,
                        batch_size: int = 32,
                        model_dir: Optional[str] = "default_model_dir",
                        **trainer_kwargs):
        """Create a new trainer instance with the loaded model weights.

        This method creates a new instance of `LightningTorchModel` and loads
        the model weights and trainer state from the specified checkpoint file.
        It restores the complete training state including model parameters,
        optimizer state, learning rate scheduler state, epoch count, step count,
        and other metadata.

        This is designed to create a new instance instead of reloading on the same
        instance to avoid shape-mismatch errors that can occur when restoring
        weights on the same instance after fitting the model, using FSDP training
        strategy.

        Note:
        This is a static method, meaning it should be called on the class directly,
        not on an instance of the class.

        Parameters
        ----------
        filepath: str
            Path to checkpoint file (.ckpt).
        model: TorchModel
            DeepChem model instance to load weights into.
        batch_size: int, default 32
            Batch size for the trainer/model.
        model_dir: str, optional (default None)
            Path to directory where model and checkpoints will be stored. If not specified,
            the default directory name lightning
        **trainer_kwargs
            Additional trainer arguments. Common options include:

            - max_epochs: int, default None
                Maximum number of training epochs.
            - accelerator: str, default "auto"
                Hardware accelerator to use ("cpu", "gpu", "tpu", "auto").
            - devices: int or str or list, default "auto"
                Number of devices/GPUs to use.
            - strategy: str, default "auto"
                Distributed training strategy ("ddp", "fsdp", "auto").
            - precision: str or int, default "32-true"
                Numerical precision ("16-mixed", "bf16-mixed", "32-true").
            - log_every_n_steps: int, default 50
                How often to log within training steps.
            - enable_checkpointing: bool, default True
                Whether to enable automatic checkpointing.
            - fast_dev_run: bool or int, default False
                Run a fast development run with limited batches for debugging.
            For all available options, see: https://lightning.ai/docs/pytorch/stable/common/trainer.html#init

        Returns
        -------
        LightningTorchModel
            New trainer instance with loaded model.

        Examples
        --------
        >>> # Call as a static method on the class
        >>> trainer = LightningTorchModel.load_checkpoint("model.ckpt", model=my_model)
        >>> # NOT: trainer.load_checkpoint("model.ckpt", model=my_model)
        """

        # Create trainer first
        trainer = LightningTorchModel(model=model,
                                      batch_size=batch_size,
                                      model_dir=model_dir,
                                      **trainer_kwargs)

        # Load the checkpoint
        trainer.lightning_model = DCLightningModule.load_from_checkpoint(
            filepath, dc_model=model)

        return trainer
