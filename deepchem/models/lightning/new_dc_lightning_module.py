import torch
import lightning as L
from deepchem.models.optimizers import LearningRateSchedule
import numpy as np


class DeepChemLightningModule(L.LightningModule):
    """
    Lightning Module for DeepChem models.
    
    Args:
        model: DeepChem model instance
    """
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=["model","dc_model"])
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
        self.uncertainty = False if not hasattr(model, 'uncertainty') else model.uncertainty

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        inputs, labels, weights = batch
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        outputs = self.model(inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if self._loss_outputs is not None:
            outputs = [outputs[i] for i in self._loss_outputs]
        loss = self._loss_fn(outputs, labels, weights)
        self.log("train_loss", loss.item(), prog_bar=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Prediction step with support for uncertainties and transformers."""
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
            if self._variance_outputs is None or len(self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs'
                )
                
        if self.output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )
                
        output_values = [t.detach().cpu().numpy() for t in output_values]

        # Apply transformers and record results
        if self.uncertainty:
            var = [output_values[i] for i in self._variance_outputs]
            if variances is None:
                variances = [var]
            else:
                for i, t in enumerate(var):
                    variances[i].append(t)
                    
        access_values = []
        if self.output_types:
            access_values += self._other_outputs
        elif self._prediction_outputs is not None:
            access_values += self._prediction_outputs

        if len(access_values) > 0:
            output_values = [output_values[i] for i in access_values]

        if len(transformers) > 0:
            if len(output_values) > 1:
                raise ValueError(
                    "predict() does not support Transformers for models with multiple outputs."
                )
            elif len(output_values) == 1:
                from deepchem.trans import undo_transforms
                output_values = [undo_transforms(output_values[0], transformers)]
                
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
        """Configure optimizers and learning rate schedulers."""
        py_optimizer = self.optimizer._create_pytorch_optimizer(
            self.model.parameters())
            
        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                py_optimizer)
            return [py_optimizer], [lr_schedule]
            
        return py_optimizer