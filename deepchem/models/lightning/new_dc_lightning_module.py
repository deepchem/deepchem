import torch
import lightning as L
from deepchem.models.optimizers import LearningRateSchedule
import numpy as np
from deepchem.models.torch_models import TorchModel, ModularTorchModel
from typing import Any, Tuple, List, Optional
from deepchem.utils.typing import OneOrMany
from deepchem.trans import Transformer


class DeepChemLightningModule(L.LightningModule):
    """A PyTorch Lightning wrapper for DeepChem models.

    This module integrates DeepChem's models with PyTorch Lightning's training loop,
    enabling efficient training and prediction workflows while managing model-specific
    operations such as loss calculation, uncertainty estimation, and data transformations.

    The class provides a consistent interface for:
      - Forward propagation through the model.
      - A training step method that computes and logs a loss value.
      - A prediction step method that handles uncertainty and additional outputs.
      - Configuration of optimizers and (optional) learning rate schedulers.

    Parameters
    ----------
    model: TorchModel
        An instance of a DeepChem TorchModel containing both
        the underlying PyTorch model and additional properties
        such as loss functions, optimizers, and output configuration.

    Examples
    --------
    Basic usage with a DeepChem TorchModel:

    >>> import torch
    >>> import lightning as L
    >>> from deepchem.models import GraphConvModel
    >>> from deepchem.models.lightning import DeepChemLightningModule
    >>> from deepchem.feat import ConvMolFeaturizer
    >>> from deepchem import data
    >>>
    >>> # Create a DeepChem model
    >>> featurizer = ConvMolFeaturizer()
    >>> model = GraphConvModel(n_tasks=1, mode='regression')
    >>>
    >>> # Wrap it in a Lightning module
    >>> lightning_module = DeepChemLightningModule(model)
    >>>
    >>> # Create a Lightning trainer
    >>> trainer = L.Trainer(max_epochs=10, accelerator='auto')
    >>>
    >>> # Prepare your data as PyTorch Lightning DataModule or DataLoader
    >>> # train_dataloader = ...  # Your training data
    >>> # val_dataloader = ...    # Your validation data
    >>>
    >>> # Train the model
    >>> # trainer.fit(lightning_module, train_dataloader, val_dataloader)
    >>>
    >>> # Make predictions
    >>> # predictions = trainer.predict(lightning_module, test_dataloader)

    Notes
    -----
    For more information, see:
      - PyTorch Lightning Documentation: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule
    """

    def __init__(self, model: TorchModel):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "dc_model"])
        self.model = model.model
        self.dc_model = model
        self.loss_mod = model.loss
        self.optimizer = model.optimizer
        self.output_types = model.output_types
        self._prediction_outputs = model._prediction_outputs
        self._loss_outputs = model._loss_outputs
        self._variance_outputs = model._variance_outputs
        self._other_outputs = model._other_outputs
        self._loss_fn = model._loss_fn
        self.uncertainty = False if not hasattr(
            model, 'uncertainty') else model.uncertainty
        self.learning_rate = model.learning_rate
        self._transformers: List[Transformer] = []
        self.other_output_types: Optional[OneOrMany[str]] = None

    def forward(self, x: Any):
        """Forward pass of the model.

        Parameters
        ----------
        x: Any
            Input data for the model.

        Returns
        -------
        Any
            Model output.
        """
        return self.model(x)

    def training_step(self, batch: Tuple[Any, Any, Any], batch_idx: int):
        """Execute a single training step, including loss computation and logging.

        The method unpacks the batch into inputs, labels, and weights and then performs:
          - A forward pass through the network.
          - Loss computation which differentiates between ModularTorchModel and
            regular TorchModel based on the provided instance.
          - Logging of the loss value for monitoring.

        Parameters
        ----------
        batch: Tuple[Any, Any, Any]
            A tuple containing:
            - inputs: data inputs to the model,
            - labels: ground truth values,
            - weights: sample weights for the loss computation.
        batch_idx: int
            Index of the current batch (useful for logging or debugging).

        Returns
        -------
        torch.Tensor
            The computed loss value as a torch tensor. This value is used for backpropagation.
        """
        inputs, labels, weights = batch
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        outputs = self.model(inputs)
        if isinstance(self.dc_model, ModularTorchModel):
            loss = self.dc_model.loss_func(inputs, labels, weights)
        elif isinstance(self.dc_model, TorchModel):
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            if self._loss_outputs is not None:
                outputs = [outputs[i] for i in self._loss_outputs]
            loss = self._loss_fn(outputs, labels, weights)
        self.log("train_loss", loss.item(), prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch: Tuple[Any, Any, Any], batch_idx: int):
        """Perform a prediction step with optional support for uncertainty estimates and data transformations.

        This method refers from _predict method form TorchModel.

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
        Any
            Model predictions, with optional uncertainty estimates if configured.
        """
        results, variances = None, None
        inputs, _, _ = batch
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        output_values = self.model(inputs)
        if isinstance(output_values, torch.Tensor):
            output_values = [output_values]

        if self.uncertainty and self.output_types:
            raise ValueError(
                'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
            )

        if self.uncertainty:
            if self._variance_outputs is None or len(
                    self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs'
                )

        if self.other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )

        output_values = [t.detach().cpu().numpy() for t in output_values]

        # Apply transformers and record results
        if self.uncertainty and self._variance_outputs is not None:
            var = [output_values[i] for i in self._variance_outputs]
            if variances is None:
                variances = [var]
            else:
                for i, t in enumerate(var):
                    variances[i].append(t)

        access_values = []
        if self.other_output_types:
            access_values += self._other_outputs
        elif self._prediction_outputs is not None:
            access_values += self._prediction_outputs

        if len(access_values) > 0:
            output_values = [output_values[i] for i in access_values]

        if len(self._transformers) > 0:
            if len(output_values) > 1:
                raise ValueError(
                    "predict() does not support Transformers for models with multiple outputs."
                )
            elif len(output_values) == 1:
                from deepchem.trans import undo_transforms
                output_values = [
                    undo_transforms(output_values[0], self._transformers)
                ]

        if results is None:
            results = [[] for i in range(len(output_values))]
        for i, t in enumerate(output_values):
            results[i].append(t)

        # Concatenate arrays to create the final results
        final_results = []
        final_variances = []
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

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Union[torch.optim.Optimizer, List]
            PyTorch optimizer or list containing optimizer and scheduler.
        """
        # No parameters to check
        py_optimizer = self.optimizer._create_pytorch_optimizer(
            self.model.parameters())

        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                py_optimizer)
            return [py_optimizer], [lr_schedule]

        return py_optimizer
