"""Model-Agnostic Meta-Learning (MAML) algorithm for low data learning."""

import os
import shutil
import tempfile
import time

from deepchem.models.optimizers import Optimizer, Adam, GradientDescent, LearningRateSchedule
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union
from deepchem.utils.typing import OneOrMany

try:
    from deepchem.metalearning import MetaLearner
    import torch
    has_pytorch = True
except:
    has_pytorch = False


class MAML(object):
    """Implements the Model-Agnostic Meta-Learning algorithm for low data learning.

    The algorithm is described in Finn et al., "Model-Agnostic Meta-Learning for Fast
    Adaptation of Deep Networks" (https://arxiv.org/abs/1703.03400).  It is used for
    training models that can perform a variety of tasks, depending on what data they
    are trained on.  It assumes you have training data for many tasks, but only a small
    amount for each one.  It performs "meta-learning" by looping over tasks and trying
    to minimize the loss on each one *after* one or a few steps of gradient descent.
    That is, it does not try to create a model that can directly solve the tasks, but
    rather tries to create a model that is very easy to train.

    To use this class, create a subclass of MetaLearner that encapsulates the model
    and data for your learning problem.  Pass it to a MAML object and call fit().
    You can then use train_on_current_task() to fine tune the model for a particular
    task.
    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn.functional as F
    >>> from deepchem.metalearning.torch_maml import MetaLearner, MAML
    >>> class SineLearner(MetaLearner):
    ...     def __init__(self):
    ...         self.batch_size = 10
    ...         self.w1 = torch.nn.Parameter(torch.tensor(np.random.normal(size=[1, 40], scale=1.0),requires_grad=True))
    ...         self.w2 = torch.nn.Parameter(torch.tensor(np.random.normal(size=[40, 40], scale=np.sqrt(1 / 40)),requires_grad=True))
    ...         self.w3 = torch.nn.Parameter(torch.tensor(np.random.normal(size=[40, 1], scale=np.sqrt(1 / 40)),requires_grad=True))
    ...         self.b1 = torch.nn.Parameter(torch.tensor(np.zeros(40)),requires_grad=True)
    ...         self.b2 = torch.nn.Parameter(torch.tensor(np.zeros(40)),requires_grad=True)
    ...         self.b3 = torch.nn.Parameter(torch.tensor(np.zeros(1)),requires_grad=True)
    ...     def compute_model(self, inputs, variables, training):
    ...         x, y = inputs
    ...         w1, w2, w3, b1, b2, b3 = variables
    ...         dense1 = F.relu(torch.matmul(x, w1) + b1)
    ...         dense2 = F.relu(torch.matmul(dense1, w2) + b2)
    ...         output = torch.matmul(dense2, w3) + b3
    ...         loss = torch.mean(torch.square(output - y))
    ...         return loss, [output]
    ...     @property
    ...     def variables(self):
    ...         return [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]
    ...     def select_task(self):
    ...         self.amplitude = 5.0 * np.random.random()
    ...         self.phase = np.pi * np.random.random()
    ...     def get_batch(self):
    ...         x = torch.tensor(np.random.uniform(-5.0, 5.0, (self.batch_size, 1)))
    ...         return [x, torch.tensor(self.amplitude * np.sin(x + self.phase))]
    ...     def parameters(self):
    ...         for key, value in self.__dict__.items():
    ...             if isinstance(value, torch.nn.Parameter):
    ...                 yield value
    >>> learner = SineLearner()
    >>> optimizer = dc.models.optimizers.Adam(learning_rate=5e-3)
    >>> maml = MAML(learner,meta_batch_size=4,optimizer=optimizer)
    >>> maml.fit(9000)

    To test it out on a new task and see how it works

    >>> learner.select_task()
    >>> maml.restore()
    >>> batch = learner.get_batch()
    >>> loss, outputs = maml.predict_on_batch(batch)
    >>> maml.train_on_current_task()
    >>> loss, outputs = maml.predict_on_batch(batch)
    """

    def __init__(
        self,
        learner: MetaLearner,
        learning_rate: Union[float, LearningRateSchedule] = 0.001,
        optimization_steps: int = 1,
        meta_batch_size: int = 10,
        optimizer: Optimizer = Adam(),
        model_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """Create an object for performing meta-optimization.

        Parameters
        ----------
        learner: MetaLearner
            defines the meta-learning problem
        learning_rate: float or Tensor
            the learning rate to use for optimizing each task (not to be confused with the one used
            for meta-learning).  This can optionally be made a variable (represented as a
            Tensor), in which case the learning rate will itself be learnable.
        optimization_steps: int
            the number of steps of gradient descent to perform for each task
        meta_batch_size: int
            the number of tasks to use for each step of meta-learning
        optimizer: Optimizer
            the optimizer to use for meta-learning (not to be confused with the gradient descent
            optimization performed for each task)
        model_dir: str
            the directory in which the model will be saved.  If None, a temporary directory will be created.
        device: torch.device, optional (default None)
            the device on which to run computations.  If None, a device is
            chosen automatically.
        """
        # Record inputs.

        self.learner: MetaLearner = learner
        self.learning_rate: Union[float, LearningRateSchedule] = learning_rate
        self.optimization_steps: int = optimization_steps
        self.meta_batch_size: int = meta_batch_size
        self.optimizer: Optimizer = optimizer

        # Create the output directory if necessary.

        self._model_dir_is_temp: bool = False
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        else:
            model_dir = tempfile.mkdtemp()
            self._model_dir_is_temp = True
        self.model_dir = model_dir
        self.save_file: str = "%s/%s" % (self.model_dir, "model")

        # Select a device.

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device: torch.device = device
        for param in learner.parameters():
            param = param.to(device)
            param.requires_grad_()

        # Create the optimizers for meta-optimization and task optimization.

        self._global_step: int = 0
        self._pytorch_optimizer = self.optimizer._create_pytorch_optimizer(
            learner.parameters())
        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            self._lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                self._pytorch_optimizer)
        else:
            self._lr_schedule = None

        task_optimizer: Optimizer = GradientDescent(
            learning_rate=self.learning_rate)
        self._pytorch_task_optimizer = task_optimizer._create_pytorch_optimizer(
            learner.parameters())
        if isinstance(task_optimizer.learning_rate, LearningRateSchedule):
            self._lr_schedule = task_optimizer.learning_rate._create_pytorch_schedule(
                self._pytorch_task_optimizer)
        else:
            self._lr_schedule = None

    def __del__(self):
        if '_model_dir_is_temp' in dir(self) and self._model_dir_is_temp:
            shutil.rmtree(self.model_dir)

    def fit(self,
            steps: int,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 600,
            restore: bool = False):
        """Perform meta-learning to train the model.

        Parameters
        ----------
        steps: int
            the number of steps of meta-learning to perform
        max_checkpoints_to_keep: int
            the maximum number of checkpoint files to keep.  When this number is reached, older
            files are deleted.
        checkpoint_interval: int
            the time interval at which to save checkpoints, measured in seconds
        restore: bool
            if True, restore the model from the most recent checkpoint before training
            it further
        """
        if restore:
            self.restore()
        checkpoint_time: float = time.time()

        # Main optimization loop.

        learner = self.learner
        variables: OneOrMany[torch.Tensor] = learner.variables
        for i in range(steps):
            self._pytorch_optimizer.zero_grad()
            for j in range(self.meta_batch_size):
                learner.select_task()
                updated_variables: OneOrMany[torch.Tensor] = variables
                for k in range(self.optimization_steps):
                    loss, _ = self.learner.compute_model(
                        learner.get_batch(), updated_variables, True)
                    gradients: Tuple[torch.Tensor, ...] = torch.autograd.grad(
                        loss,
                        updated_variables,
                        grad_outputs=torch.ones_like(loss),
                        create_graph=True,
                        retain_graph=True)
                    updated_variables = [
                        v if g is None else v - self.learning_rate * g
                        for v, g in zip(updated_variables, gradients)
                    ]
                meta_loss, _ = self.learner.compute_model(
                    learner.get_batch(), updated_variables, True)
                meta_gradients: Tuple[torch.Tensor, ...] = torch.autograd.grad(
                    meta_loss,
                    variables,
                    grad_outputs=torch.ones_like(meta_loss),
                    retain_graph=True)
                if j == 0:
                    summed_gradients: Union[Tuple[torch.Tensor, ...],
                                            List[torch.Tensor]] = meta_gradients
                else:
                    summed_gradients = [
                        s + g for s, g in zip(summed_gradients, meta_gradients)
                    ]
                ind: int = 0
                for param in self.learner.parameters():
                    param.grad = summed_gradients[ind]
                    ind = ind + 1

            self._pytorch_optimizer.step()
            if self._lr_schedule is not None:
                self._lr_schedule.step()

            # Do checkpointing.

            if i == steps - 1 or time.time(
            ) >= checkpoint_time + checkpoint_interval:
                self.save_checkpoint(max_checkpoints_to_keep)
                checkpoint_time = time.time()

    def restore(self) -> None:
        """Reload the model parameters from the most recent checkpoint file."""
        last_checkpoint: Union[List[str], str] = sorted(
            self.get_checkpoints(self.model_dir))
        if len(last_checkpoint) == 0:
            raise ValueError('No checkpoint found')
        last_checkpoint = last_checkpoint[0]
        data: Any = torch.load(last_checkpoint, map_location=self.device)
        self.learner.__dict__ = data['model_state_dict']
        self.learning_rate = data['learning_rate']
        self._pytorch_optimizer.load_state_dict(data['optimizer_state_dict'])
        self._pytorch_task_optimizer.load_state_dict(
            data['task_optimizer_state_dict'])
        self._global_step = data['global_step']

    def train_on_current_task(self,
                              optimization_steps: int = 1,
                              restore: bool = True):
        """Perform a few steps of gradient descent to fine tune the model on the current task.

        Parameters
        ----------
        optimization_steps: int
            the number of steps of gradient descent to perform
        restore: bool
            if True, restore the model from the most recent checkpoint before optimizing
        """
        if restore:
            self.restore()
        variables: OneOrMany[torch.Tensor] = self.learner.variables
        task_optimizer: Optimizer = GradientDescent(
            learning_rate=self.learning_rate)
        self._pytorch_task_optimizer = task_optimizer._create_pytorch_optimizer(
            self.learner.parameters())
        if isinstance(task_optimizer.learning_rate, LearningRateSchedule):
            self._lr_schedule = task_optimizer.learning_rate._create_pytorch_schedule(
                self._pytorch_task_optimizer)
        else:
            self._lr_schedule = None
        for i in range(optimization_steps):
            self._pytorch_task_optimizer.zero_grad()
            inputs = self.learner.get_batch()
            loss, _ = self.learner.compute_model(inputs, variables, True)
            loss.backward()
            self._pytorch_task_optimizer.step()

    def predict_on_batch(
        self, inputs: OneOrMany[torch.Tensor]
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        """Compute the model's outputs for a batch of inputs.

        Parameters
        ----------
        inputs: list of arrays
            the inputs to the model

        Returns
        -------
        (loss, outputs) where loss is the value of the model's loss function, and
        outputs is a list of the model's outputs
        """
        return self.learner.compute_model(inputs, self.learner.variables, False)

    def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 5,
                        model_dir: Optional[str] = None) -> None:
        """Save a checkpoint to disk.

        Usually you do not need to call this method, since fit() saves checkpoints
        automatically.  If you have disabled automatic checkpointing during fitting,
        this can be called to manually write checkpoints.

        Parameters
        ----------
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        model_dir: str, default None
            Model directory to save checkpoint to. If None, revert to self.model_dir
        """
        if model_dir is None:
            model_dir = self.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the checkpoint to a file.

        data: Dict[str, Any] = {
            'model_state_dict':
                self.learner.__dict__,
            'learning_rate':
                self.learning_rate,
            'optimizer_state_dict':
                self._pytorch_optimizer.state_dict(),
            'task_optimizer_state_dict':
                self._pytorch_task_optimizer.state_dict(),
            'global_step':
                self._global_step
        }
        temp_file: str = os.path.join(model_dir, 'temp_checkpoint.pt')
        torch.save(data, temp_file)

        # Rename and delete older files.

        paths: List[str] = [
            os.path.join(model_dir, 'checkpoint%d.pt' % (i + 1))
            for i in range(max_checkpoints_to_keep)
        ]
        if os.path.exists(paths[-1]):
            os.remove(paths[-1])
        for i in reversed(range(max_checkpoints_to_keep - 1)):
            if os.path.exists(paths[i]):
                os.rename(paths[i], paths[i + 1])
        os.rename(temp_file, paths[0])

    def get_checkpoints(self, model_dir: Optional[str] = None) -> List[str]:
        """Get a list of all available checkpoint files.

        Parameters
        ----------
        model_dir: str, default None
            Directory to get list of checkpoints from. Reverts to self.model_dir if None

        """
        if model_dir is None:
            model_dir = self.model_dir
        files: List[str] = sorted(os.listdir(model_dir))
        files = [
            f for f in files if f.startswith('checkpoint') and f.endswith('.pt')
        ]
        return [os.path.join(model_dir, f) for f in files]
