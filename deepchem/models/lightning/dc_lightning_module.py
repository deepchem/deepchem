import torch
import lightning as L  # noqa
from deepchem.models.torch_models import ModularTorchModel, TorchModel
import numpy as np
from deepchem.utils.typing import List, OneOrMany, Any, LossFn
from typing import Optional, Union, Tuple
from deepchem.trans import Transformer, undo_transforms
from deepchem.models.optimizers import LearningRateSchedule, Optimizer


class DCLightningModule(L.LightningModule):
    """DeepChem Lightning Module to be used with Lightning trainer.

    The lightning module is a wrapper over deepchem's torch model.
    This module directly works with pytorch lightning trainer
    which runs training for multiple epochs and also is responsible
    for setting up and training models on multiple GPUs.
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule

    Examples
    --------
    Training and prediction workflow with a GCN model:

    >>> import deepchem as dc
    >>> import lightning as L
    >>> from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule
    >>> from deepchem.models.lightning.dc_lightning_module import DCLightningModule
    >>> from deepchem.feat import MolGraphConvFeaturizer
    >>>
    >>> # Load and prepare dataset
    >>> tasks, dataset, transformers = dc.molnet.load_bace_classification(
    ...     featurizer=MolGraphConvFeaturizer(), reload=False)
    >>>
    >>> # Create a GCN model
    >>> model = dc.models.GCNModel(
    ...     mode='classification',
    ...     n_tasks=len(tasks),
    ...     number_atom_features=30,
    ...     batch_size=10,
    ...     learning_rate=0.0003
    ... )
    >>>
    >>> # Setup Lightning modules
    >>> data_module = DCLightningDatasetModule(
    ...     dataset=dataset[0],
    ...     batch_size=10,
    ...     model=model
    ... )
    >>> lightning_model = DCLightningModule(dc_model=model)
    >>>
    >>> # Setup trainer and fit
    >>> trainer = L.Trainer(
    ...     fast_dev_run=True,
    ...     accelerator="auto",
    ...     devices="auto",
    ...     logger=False,
    ...     enable_checkpointing=True
    ... )
    >>> # trainer.fit(model=lightning_model, datamodule=data_module)
    >>>
    >>> # Make predictions
    >>> # prediction_batches = trainer.predict(model=lightning_model, datamodule=data_module)

    Notes
    -----
    This class requires PyTorch to be installed.
    """

    def __init__(self, dc_model):
        """Create a new DCLightningModule.

        Parameters
        ----------
        dc_model: TorchModel or ModularTorchModel
            TorchModel to be wrapped inside the lightning module.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['dc_model'])
        self.dc_model = dc_model
        self.pt_model = dc_model.model
        self.loss: LossFn = dc_model._loss_fn
        self.optimizer: Optimizer = dc_model.optimizer
        self._prediction_outputs: Optional[
            List[int]] = dc_model._prediction_outputs
        self._loss_outputs: Optional[List[int]] = dc_model._loss_outputs
        self._variance_outputs: Optional[List[int]] = dc_model._variance_outputs
        self._other_outputs: Optional[List[int]] = dc_model._other_outputs
        self.uncertainty: bool = getattr(dc_model, 'uncertainty', False)
        self.learning_rate: Union[float,
                                  LearningRateSchedule] = dc_model.learning_rate
        self.transformers: List[Transformer] = []
        self.other_output_types: Optional[OneOrMany[str]] = None

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]]
            PyTorch optimizer or tuple containing lists of optimizers and schedulers.
        """
        self.dc_model._built = True
        pytorch_optimizer = self.optimizer._create_pytorch_optimizer(
            self.pt_model.parameters())

        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                pytorch_optimizer)
            return [pytorch_optimizer], [lr_schedule]

        return pytorch_optimizer

    def training_step(self, batch: Union[Tuple[Any, Any, Any], Any],
                      batch_idx: int):
        """Perform a training step.

        Parameters
        ----------
        batch: Union[Tuple[Any, Any, Any], Any]
            A tensor, tuple or list containing inputs, labels, and weights.
        batch_idx: int
            Integer displaying index of this batch

        Returns
        -------
        torch.Tensor
            The computed loss tensor for this training step.
        """
        if hasattr(batch, 'batch_list'):
            # If batch is a DCLightningDatasetBatch, extract the batch list.
            batch_list = batch.batch_list  # type: ignore
            inputs, labels, weights = self.dc_model._prepare_batch(batch_list)
        else:
            inputs, labels, weights = batch

        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]

        if isinstance(self.dc_model, ModularTorchModel):
            loss = self.dc_model.loss_func(inputs, labels, weights)
        elif isinstance(self.dc_model, TorchModel):
            outputs = self.pt_model(inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]

            if self.dc_model._loss_outputs is not None:
                outputs = [outputs[i] for i in self.dc_model._loss_outputs]
            loss = self.loss(outputs, labels, weights)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
            batch_size=self.dc_model.batch_size,
        )

        return loss

    def predict_step(self, batch: Tuple[Any, Any, Any], batch_idx: int):
        """Perform a prediction step with optional support for uncertainty estimates and data transformations.

        This method was copied from TorchModel._predict and adapted for Lightning's predict_step interface.

        Changes include:
        - removed the `self.dc_model._prepare_batch` call since `batch` is already prepared.

        Parameters
        ----------
        batch: Tuple[Any, Any, Any]
            A tuple containing:
            - inputs: the input data for prediction,
            - labels: (unused in prediction, but maintained for consistency),
            - weights: (unused in prediction).
        batch_idx: int
            Index of the current batch.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray], zip]
            Model predictions for this batch. Can be:
            - numpy array for single output models
            - list of numpy arrays for multi-output models
            - zip of (predictions, variances) if uncertainty is enabled
        """
        results: Optional[List[List[np.ndarray]]] = None
        variances: Optional[List[List[np.ndarray]]] = None
        if self.uncertainty and (self.other_output_types is not None):
            raise ValueError(
                'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
            )
        if self.uncertainty:
            if self._variance_outputs is None or len(
                    self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if self._prediction_outputs is None or len(
                    self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs'
                )
        if self.other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )
        inputs, _, _ = batch
        # Invoke the model.
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        output_values: Union[torch.Tensor,
                             List[torch.Tensor]] = self.pt_model(inputs)
        if isinstance(output_values, torch.Tensor):
            output_values = [output_values]
        output_values_np: List[np.ndarray] = [
            t.detach().cpu().numpy() for t in output_values
        ]

        # Apply tranformers and record results.
        if self.uncertainty:
            if self._variance_outputs is not None:
                var: List[np.ndarray] = [
                    output_values_np[i] for i in self._variance_outputs
                ]
                if variances is None:
                    variances = [var]
                else:
                    for i, t in enumerate(var):
                        variances[i].append(t)
        access_values: List[int] = []
        if self.other_output_types:
            if self._other_outputs is not None:
                access_values += self._other_outputs
        elif self._prediction_outputs is not None:
            access_values += self._prediction_outputs

        if len(access_values) > 0:
            output_values_np = [output_values_np[i] for i in access_values]

        if len(self.transformers) > 0:
            if len(output_values_np) > 1:
                raise ValueError(
                    "predict() does not support Transformers for models with multiple outputs."
                )
            elif len(output_values_np) == 1:
                output_values_np = [
                    undo_transforms(output_values_np[0], self.transformers)
                ]
        if results is None:
            results = [[] for i in range(len(output_values_np))]
        for i, t in enumerate(output_values_np):
            results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results: List[np.ndarray] = []
        final_variances: List[np.ndarray] = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))
        if self.uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances)
        if len(final_results) == 1:
            return final_results[0]
        else:
            return final_results
