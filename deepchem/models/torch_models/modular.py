import torch 
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel
from typing import Optional
from deepchem.models.optimizers import LearningRateSchedule
import time
import logging
from collections.abc import Sequence as SequenceCollection
from typing import Any, Callable, Iterable, List, Optional,  Tuple, Union
from deepchem.utils.typing import LossFn, OneOrMany
import torch
logger = logging.getLogger(__name__)

class ModularTorchModel(TorchModel):
    def __init__(self, model:nn.Module, components:dict, **kwargs):
        self.model = model
        self.components = components
        super().__init__(self.model, self.loss_func, **kwargs)
        # send self.model and self.components to the device
        self.model.to(self.device)
        self.components = {k: v.to(self.device) for k, v in self.components.items()}
    
    def build_model(self):
        return NotImplementedError("Subclass must define the components")
    
    def build_components(self):
        return NotImplementedError("Subclass must define the components")   
    
    def loss_func(self):
        return NotImplementedError("Subclass must define the loss function")
    
    def load_from_pretrained(self, source_model: 'ModularTorchModel' = None, checkpoint: Optional[str] = None, model_dir: str = None, components: list = None):
        # generate the source state dict
        if source_model is not None:
            source_state_dict = source_model.model.state_dict()
        elif checkpoint is not None:
            source_state_dict = torch.load(checkpoint)['model_state_dict']
        elif model_dir is not None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            source_state_dict = torch.load(checkpoints[0])['model_state_dict']
        else:
            raise ValueError("Must provide a source model, checkpoint, or model_dir")
    
        if components is not None: # load the specified components
            if source_model is not None:
                assignment_map = {k: v for k, v in source_model.components.items() if k in components}
                self.components.update(assignment_map)
                self.model = self.build_model()
            else:                
                raise ValueError("If loading from checkpoint, you cannot pass a list of components to load")
        else: # or all components with matching names and shapes
            model_dict = self.model.state_dict()
            assignment_map = {k: v for k, v in source_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape} 
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
        """Train this model on data from a generator.
    
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

            avg_loss += batch_loss

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
    