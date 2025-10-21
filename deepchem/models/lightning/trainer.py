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
    and checkpoint management. Currently, it supports strategies like DDP (Distributed Data Parallel)
    and FSDP (Fully Sharded Data Parallel) for distributed training, as well as single-device training.

    **Important**: For multi-GPU strategies (DDP, FSDP), this class must be used in a script
    and cannot be run in Jupyter notebooks or interactive environments due to Lightning's
    multiprocessing requirements.
    """

    def __init__(self,
                 model: TorchModel,
                 batch_size: int = 32,
                 model_dir: Optional[str] = "default_model_dir",
                 **trainer_kwargs: Any) -> None:
        """Initialize the LightningTorchModel.

        Parameters
        ----------
        model: TorchModel
            Initialized DeepChem model to be trained or used for inference.
        batch_size: int, default 32
            Batch size for training and prediction data loaders.
        model_dir: str, default "default_model_dir"
            Path to directory where model and checkpoints will be stored. If not specified,
            model will be stored in a "default_model_dir" directory. This is compatible with
            DeepChem's model directory structure. If None, checkpointing will be disabled.
        **trainer_kwargs
            Additional keyword arguments passed to the Lightning Trainer. Common options include:

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
                Run a fast development run with limited batches, epochs and no checkpointing for debugging.
            For all available options, see: https://lightning.ai/docs/pytorch/stable/common/trainer.html#init

        Examples
        --------
        >>> import deepchem as dc
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
        ...     accelerator="cpu",
        ...     log_every_n_steps=1,
        ...     fast_dev_run=True
        ... )
        >>> # Train with custom checkpoint settings
        >>> # trainer.fit(valid_dataset, nb_epoch=3)
        >>> # predictions = trainer.predict(valid_dataset)
        >>> # To restore from checkpoint:
        >>> # trainer.restore()
        """
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
            nb_epoch: int = 1,
            restore: bool = False,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 1000,
            num_workers: int = 4,
            ckpt_path: Optional[str] = None):
        """Train the model on the provided dataset.

        Parameters
        ----------
        train_dataset: dc.data.Dataset
            DeepChem dataset for training.
        nb_epoch: int, default 1
            Maximum number of epochs to train the model for.
            Note, nb_epoch is mapped to `max_epochs` in Lightning Trainer.
        restore: bool, default False
            Whether to restore from a previous checkpoint. If True, will load the model weights
            from the specified `ckpt_path` if provided. If `restore` is True and `ckpt_path` is None,
            it will look for the last checkpoint in the `model_dir` under "checkpoints/last.ckpt".
        max_checkpoints_to_keep: int, default 5
            The maximum number of checkpoints to keep.
        checkpoint_interval: int, default 1000
            The frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        num_workers: int, default 4
            Number of workers for DataLoader.
        ckpt_path: Optional[str], default None
            Path to a checkpoint file to resume training from. If None, starts fresh.

        Notes
        -----
        If `max_checkpoints_to_keep` is set to n, the trainer will keep the last n checkpoints plus the
        last checkpoint created when the fit ends successfully, named `last.ckpt`.
        """

        self.trainer_kwargs['max_epochs'] = nb_epoch

        # If restore is True, we need to check if ckpt_path is provided
        if restore and ckpt_path is None:
            self.restore()

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
                predictions = np.concatenate([  # type: ignore
                    p for p in predictions
                ])
            except ValueError:
                # If concatenation fails (e.g., due to different shapes in MLM tasks etc),
                # return the predictions as a list of batches directly
                pass
        else:
            predictions = []
        return predictions

    def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 1,
                        model_dir: Optional[str] = None) -> None:
        """Save a checkpoint to disk.

        Usually you do not need to call this method, since fit() saves checkpoints
        automatically. If you have disabled automatic checkpointing during fitting,
        this can be called to manually write checkpoints.

        This method maintains compatibility with TorchModel's save_checkpoint interface
        while using Lightning's native checkpointing mechanism.

        Parameters
        ----------
        max_checkpoints_to_keep : int, default 1
            The maximum number of checkpoints to keep. Older checkpoints are discarded.
        model_dir : str, default None
            Model directory to save checkpoint to. If None, reverts to self.model_dir.
            Checkpoints will be saved in a 'checkpoints' subdirectory within this path.

        Notes
        -----
        The `max_checkpoints_to_keep` parameter greater than `1` does not play any
        significant role here, since we use modelcheckpoint callbacks from lightning for dynamic checkpoint
        saving. It is kept with the same name and type just to follow the deepchem's convention.
        """
        if max_checkpoints_to_keep == 0:
            return

        if model_dir is None:
            model_dir = self.model_dir

        # Check if trainer has been initialized
        if not hasattr(self, 'trainer') or self.trainer is None:
            raise RuntimeError(
                "Trainer has not been initialized. Please call fit() or predict() "
                "before attempting to save a checkpoint manually.")

        # Create checkpoints subdirectory following ModelCheckpoint convention
        checkpoints_dir = os.path.join(model_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Save as 'last_manual.ckpt' to match ModelCheckpoint convention
        checkpoint_path = os.path.join(checkpoints_dir, 'last_manual.ckpt')
        self.trainer.save_checkpoint(checkpoint_path)

    def restore(self,
                checkpoint: Optional[str] = None,
                model_dir: Optional[str] = None,
                strict: Optional[bool] = True) -> None:
        """Reload the values of all variables from a checkpoint file.

        This method maintains compatibility with TorchModel's restore interface
        while using Lightning's native checkpointing mechanism.

        Parameters
        ----------
        checkpoint: str, optional
            the path to the checkpoint file to load. If this is None, will look for
            'last.ckpt' in the model_dir/checkpoints/ directory.
        model_dir: str, default None
            Directory to restore checkpoint from. If None, use self.model_dir.  If
            checkpoint is not None, this is ignored.
        strict: bool, default True
            Whether to strictly enforce that the keys in the checkpoint, match the keys
            returned by this module's state dict.

        Notes
        -----
        **Important Note for FSDP Users**: When using FSDP (Fully Sharded Data Parallel)
        training strategy, restoring weights on the same trainer instance after fitting,
        for prediction, can cause shape-mismatch errors due to how FSDP handles model sharding.
        **It is strongly recommended to create a new LightningTorchModel instance**
        instead of calling restore() on an existing trained instance when using FSDP.
        """
        logger.info('Restoring model')

        if checkpoint is None:
            # Look for the default checkpoint location
            if model_dir is None:
                model_dir = self.model_dir

            # Check for last.ckpt in checkpoints subdirectory
            checkpoints_dir = os.path.join(model_dir, "checkpoints")
            checkpoint_path = os.path.join(checkpoints_dir, "last.ckpt")
            if os.path.exists(checkpoint_path):
                checkpoint = checkpoint_path
            else:
                # Look for any .ckpt file in the model directory
                if os.path.exists(checkpoints_dir):
                    ckpt_files = [
                        f for f in os.listdir(checkpoints_dir)
                        if f.endswith('.ckpt')
                    ]
                    if ckpt_files:
                        checkpoint = os.path.join(checkpoints_dir,
                                                  sorted(ckpt_files)[0])
                    else:
                        raise ValueError(
                            f'No checkpoint found in {checkpoints_dir}')
                else:
                    raise ValueError(
                        f'Model directory {checkpoints_dir} does not exist')

        # Load the checkpoint using Lightning's mechanism
        self.lightning_model = DCLightningModule.load_from_checkpoint(
            checkpoint, dc_model=self.model, strict=strict)
