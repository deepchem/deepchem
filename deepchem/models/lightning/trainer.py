from deepchem.data import Dataset
from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule
from deepchem.models.lightning.dc_lightning_module import DCLightningModule
from deepchem.models.torch_models import TorchModel
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.evaluate import _process_metric_input, Score, Metrics
from deepchem.metrics import Metric
from deepchem.utils.typing import OneOrMany
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import numpy as np
import lightning as L
from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')
logger = logging.getLogger(__name__)


class LightningTorchModel:
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
    >>> trainer.fit(valid_dataset)
    >>> predictions = trainer.predict(valid_dataset)
    >>> trainer.save_checkpoint("model.ckpt")
    >>> # To reload:
    >>> trainer2 = LightningTorchModel.load_checkpoint("model.ckpt", model=model)
    """

    def __init__(self,
                 model: TorchModel,
                 batch_size: int = 32,
                 **trainer_kwargs: Any) -> None:
        self.model: TorchModel = model
        self.batch_size: int = batch_size
        self.trainer_kwargs: Dict[str, Any] = trainer_kwargs
        self.trainer: L.Trainer = L.Trainer(**self.trainer_kwargs)
        # Create the Lightning module
        self.lightning_model: DCLightningModule = DCLightningModule(model)

    def fit(self,
            train_dataset: Dataset,
            num_workers: int = 4,
            ckpt_path: Optional[str] = None):
        """Train the model on the provided dataset.

        Parameters
        ----------
        train_dataset: dc.data.Dataset
            DeepChem dataset for training.
        num_workers: int, default 4
            Number of workers for DataLoader.
        ckpt_path: Optional[str], default None
            Path to a checkpoint file to resume training from. If None, starts fresh.
        """

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

        return predictions

    def evaluate(self,
                 dataset: Dataset,
                 metrics: Metrics,
                 transformers: List[Transformer] = [],
                 per_task_metrics: bool = False,
                 use_sample_weights: bool = False,
                 n_classes: int = 2) -> Union[Score, Tuple[Score, Score]]:
        """
        Evaluate model performance on a dataset using Lightning for multi-GPU support.

        Changes compared to the original `evaluate` method:
        - Uses `LightningTorchModel's` predict method to get predictions.
        - Performs additional concatenation of predictions to ensure correct shape.

        This method refers to the `evaluate` method in the `Evaluator` class

        Parameters
        ----------
        dataset: Dataset
            DeepChem dataset to evaluate on.
        metrics: Metrics
            The set of metrics to compute. Can be a single metric, list of metrics,
            or metric functions.
        transformers: List[Transformer], default []
            List of transformers that were applied to the dataset.
        per_task_metrics: bool, default False
            If true, return computed metric for each task on multitask dataset.
        use_sample_weights: bool, default False
            If set, use per-sample weights.
        n_classes: int, default 2
            Number of unique classes for classification metrics.

        Returns
        -------
        Union[Score, Tuple[Score, Score]]
            Dictionary mapping metric names to scores. If per_task_metrics is True,
            returns a tuple of (multitask_scores, all_task_scores).
        """

        # Process input metrics
        processed_metrics: List[Metric] = _process_metric_input(metrics)

        y: np.ndarray = dataset.y
        w: np.ndarray = dataset.w

        output_transformers = [t for t in transformers if t.transform_y]
        y = undo_transforms(y, output_transformers)

        # Get predictions using Lightning's predict
        y_pred = self.predict(dataset, transformers=output_transformers)
        if y_pred is None:
            raise ValueError("Prediction failed - no results returned")
        y_pred = np.concatenate([p for p in y_pred])

        n_tasks = len(dataset.get_task_names())

        multitask_scores = {}
        all_task_scores = {}

        # Compute metrics
        for metric in processed_metrics:
            results = metric.compute_metric(
                y,
                y_pred,
                w,
                per_task_metrics=per_task_metrics,
                n_tasks=n_tasks,
                n_classes=n_classes,
                use_sample_weights=use_sample_weights)

            if per_task_metrics:
                multitask_scores[metric.name], computed_metrics = results
                all_task_scores[metric.name] = computed_metrics
            else:
                multitask_scores[metric.name] = results

        if not per_task_metrics:
            return multitask_scores
        else:
            return multitask_scores, all_task_scores

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
                        **trainer_kwargs):
        """Create a new trainer instance with the loaded model weights.

        This method creates a new instance of `LightningTorchModel` and loads
        the model weights and trainer state from the specified checkpoint file.
        It restores the complete training state including model parameters,
        optimizer state, learning rate scheduler state, epoch count, step count,
        and other metadata.

        This is designed to create a new instance instead of reloading on the same
        instance to avoid shape-mismatch errors that can occur when restoring
        weights on the same instance after fitting the model, using FSDP.

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
        **trainer_kwargs
            Additional trainer arguments.
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
                                      **trainer_kwargs)

        # Load the checkpoint
        trainer.lightning_model = DCLightningModule.load_from_checkpoint(
            filepath, dc_model=model)

        return trainer
