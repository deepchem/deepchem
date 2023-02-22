import time
import logging
import copy
from collections.abc import Sequence as SequenceCollection
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.utils.typing import LossFn, OneOrMany

logger = logging.getLogger(__name__)


class ModularTorchModel(TorchModel):
    """ModularTorchModel is a subclass of TorchModel that allows for components to be
    pretrained and then combined into a final model. It is designed to be subclassed
    for specific models and is not intended to be used directly. There are 3 main differences
    between ModularTorchModel and TorchModel:

    - The build_components() method is used to define the components of the model.
    - The components are combined into a final model with the build_model() method.
    - The loss function is defined with the loss_func method. This may access the
      components to compute the loss using intermediate values from the network, rather
      than just the full forward pass output.

    Here is an example of how to use ModularTorchModel to pretrain a linear layer, load
    it into another network and then finetune that network:

    >>> import numpy as np
    >>> import deepchem as dc
    >>> import torch
    >>> n_samples = 6
    >>> n_feat = 3
    >>> n_hidden = 2
    >>> n_tasks = 6
    >>> pt_tasks = 3
    >>> X = np.random.rand(n_samples, n_feat)
    >>> y_pretrain = np.zeros((n_samples, pt_tasks)).astype(np.float32)
    >>> dataset_pt = dc.data.NumpyDataset(X, y_pretrain)
    >>> y_finetune = np.zeros((n_samples, n_tasks)).astype(np.float32)
    >>> dataset_ft = dc.data.NumpyDataset(X, y_finetune)
    >>> components = {'linear': torch.nn.Linear(n_feat, n_hidden),
    ... 'activation': torch.nn.ReLU(), 'head': torch.nn.Linear(n_hidden, n_tasks)}
    >>> model = torch.nn.Sequential(components['linear'], components['activation'],
    ... components['head'])
    >>> modular_model = dc.models.torch_models.modular.ModularTorchModel(model, components)
    >>> def example_loss_func(inputs, labels, weights):
    ...    return (torch.nn.functional.mse_loss(model(inputs), labels[0]) * weights[0]).mean()
    >>> modular_model.loss_func = example_loss_func
    >>> def example_model_build():
    ...     return torch.nn.Sequential(components['linear'], components['activation'],
    ... components['head'])
    >>> modular_model.build_model = example_model_build
    >>> pretrain_components = {'linear': torch.nn.Linear(n_feat, n_hidden),
    ... 'activation': torch.nn.ReLU(), 'head': torch.nn.Linear(n_hidden, pt_tasks)}
    >>> pretrain_model = torch.nn.Sequential(pretrain_components['linear'],
    ... pretrain_components['activation'], pretrain_components['head'])
    >>> pretrain_modular_model = dc.models.torch_models.modular.ModularTorchModel(pretrain_model,
    ... pretrain_components)
    >>> def example_pt_loss_func(inputs, labels, weights):
    ...     return (torch.nn.functional.mse_loss(pretrain_model(inputs), labels[0]) * weights[0]).mean()
    >>> pretrain_modular_model.loss_func = example_pt_loss_func
    >>> pt_loss = pretrain_modular_model.fit(dataset_pt, nb_epoch=1)
    >>> modular_model.load_pretrained_components(pretrain_modular_model, components=['linear'])
    >>> ft_loss = modular_model.fit(dataset_ft, nb_epoch=1)

    """

    def __init__(self, model: nn.Module, components: dict, **kwargs):
        """Create a ModularTorchModel.

        Parameters
        ----------
        model: nn.Module
            The model to be trained.
        components: dict
            A dictionary of the components of the model. The keys are the names of the
            components and the values are the components themselves.
        """

        self.model = model
        self.components = components
        super().__init__(self.model, self.loss_func, **kwargs)
        self.model.to(self.device)
        self.components = {
            k: v.to(self.device) for k, v in self.components.items()
        }

    def build_model(self) -> nn.Module:
        """Builds the final model from the components."""
        raise NotImplementedError("Subclass must define the components")

    def build_components(self) -> dict:
        """Creates the components dictionary, with the keys being the names of the
        components and the values being torch.nn.module objects."""
        raise NotImplementedError("Subclass must define the components")

    def loss_func(self, inputs: OneOrMany[torch.Tensor], labels: Sequence,
                  weights: Sequence) -> torch.Tensor:
        """Defines the loss function for the model which can access the components
        using self.components. The loss function should take the inputs, labels, and
        weights as arguments and return the loss."""
        raise NotImplementedError("Subclass must define the loss function")

    def freeze_components(self, components: List[str]):
        """Freezes or unfreezes the parameters of the specified components.

        Components string refers to keys in self.components.

        Parameters
        ----------
        components: List[str]
            The components to freeze.
        """
        for component in components:
            for param in self.components[component].parameters():
                param.requires_grad = False

    def unfreeze_components(self, components: List[str]):
        """Unfreezes the parameters of the specified components.

        Components string refers to keys in self.components.

        Parameters
        ----------
        components: List[str]
            The components to unfreeze.
        """
        for component in components:
            for param in self.components[component].parameters():
                param.requires_grad = True

    def load_pretrained_components(
            self,
            source_model: Optional['ModularTorchModel'] = None,
            checkpoint: Optional[str] = None,
            model_dir: Optional[str] = None,
            components: Optional[list] = None) -> None:
        """Modifies the TorchModel load_from_pretrained method to allow for loading
        from a ModularTorchModel and specifying which components to load.

        If the user does not a specify a source model, a checkpoint is used to load
        the weights. In this case, the user cannot specify which components to load
        because the components are not stored in the checkpoint. All layers will
        then be loaded if they have the same name and shape. This can cause issues
        if a pretrained model has similar but not identical layers to the model where
        a user may expect the weights to be loaded. ModularTorchModel subclasses
        should be written such that the components are atomic and will be preserved
        across as many tasks as possible. For example, an encoder may have varying
        input dimensions for different datasets, so the encoder should be written
        such that the input layer is not included in the encoder, allowing the
        encoder to be loaded with any input dimension.

        Parameters
        ----------
        source_model: Optional[ModularTorchModel]
            The model to load the weights from.
        checkpoint: Optional[str]
            The path to the checkpoint to load the weights from.
        model_dir: Optional[str]
            The path to the directory containing the checkpoint to load the weights.
        components: Optional[list]
            The components to load the weights from. If None, all components will be
            loaded.
        """

        # generate the source state dict
        if source_model is not None:
            source_state_dict = source_model.model.state_dict()
        elif checkpoint is not None:
            source_state_dict = torch.load(checkpoint)['model_state_dict']
        elif model_dir is not None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            source_state_dict = torch.load(checkpoints[0])['model_state_dict']
        else:
            raise ValueError(
                "Must provide a source model, checkpoint, or model_dir")

        if components is not None:  # load the specified components
            if source_model is not None:
                assignment_map = {
                    k: v
                    for k, v in source_model.components.items()
                    if k in components
                }
                assignment_map_copy = copy.deepcopy(
                    assignment_map)  # deep copy to avoid modifying source_model
                self.components.update(assignment_map_copy)
                self.model = self.build_model()
            else:
                raise ValueError(
                    "If loading from checkpoint, you cannot pass a list of components to load"
                )
        else:  # or all components with matching names and shapes
            model_dict = self.model.state_dict()
            assignment_map = {
                k: v
                for k, v in source_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(assignment_map)
            self.model.load_state_dict(model_dict)

    def fit_generator(self,
                      generator: Iterable[Tuple[Any, Any, Any]],
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False,
                      variables: Optional[List[torch.nn.Parameter]] = None,
                      loss: Optional[LossFn] = None,
                      callbacks: Union[Callable, List[Callable]] = [],
                      all_losses: Optional[List[float]] = None) -> float:
        """Train this model on data from a generator. This method is similar to
        the TorchModel implementation, but it passes the inputs directly to the
        loss function, rather than passing them through the model first.  This
        enables the loss to be calculated from intermediate steps of the model
        and not just the final output.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: list of torch.nn.Parameter
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step) that will be invoked after
            every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.

        Returns
        -------
        The average loss over the most recent checkpoint interval
        """

        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if loss is None:
            loss = self._loss_fn
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate,
                              LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        # Main training loop.

        for batch in generator:
            if restore:
                self.restore()
                restore = False
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self._prepare_batch(batch)

            # Execute the loss function, accumulating the gradients.

            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]

            optimizer.zero_grad()
            batch_loss = self.loss_func(inputs, labels, weights)
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step

            avg_loss += float(batch_loss)

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info('Ending global_step %d: Average loss %g' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        if checkpoint_interval > 0:
            self.save_checkpoint(max_checkpoints_to_keep)

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss
