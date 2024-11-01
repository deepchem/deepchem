import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple, Callable, Literal, Union
from typing import Sequence as SequenceCollection
from deepchem.utils.typing import OneOrMany, ActivationFn

logger = logging.getLogger(__name__)


class RobustMultitask(nn.Module):
    """Implements a neural network for robust multitasking.

    The key idea of this model is to have bypass layers that feed
    directly from features to task output. This might provide some
    flexibility toroute around challenges in multitasking with
    destructive interference.

    References
    ----------
    This technique was introduced in [1]_

    .. [1] Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

    """

    def __init__(self,
                 n_tasks: int,
                 n_features: int,
                 layer_sizes: SequenceCollection[int] = [1000],
                 mode: Literal['regression', 'classification'] = 'regression',
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 weight_decay_penalty=0.0,
                 weight_decay_penalty_type="l2",
                 activation_fns: OneOrMany[ActivationFn] = nn.ReLU(),
                 dropouts: OneOrMany[float] = 0.5,
                 n_classes: int = 2,
                 bypass_layer_sizes: SequenceCollection[int] = [100],
                 bypass_weight_init_stddevs: OneOrMany[float] = [.02],
                 bypass_bias_init_consts: OneOrMany[float] = [1.0],
                 bypass_dropouts: OneOrMany[float] = [0.5],
                 **kwargs):

        self.n_tasks: int = n_tasks
        self.n_features: int = n_features
        self.n_classes: int = n_classes
        self.mode: Literal['regression', 'classification'] = mode
        self.layer_sizes: SequenceCollection[int] = layer_sizes
        self.bypass_layer_sizes: SequenceCollection[int] = ([
            bypass_layer_sizes
        ] if isinstance(bypass_layer_sizes, int) else bypass_layer_sizes)
        self.weight_decay_penalty: float = weight_decay_penalty
        self.weight_decay_penalty_type: Literal[
            'l1', 'l2'] = weight_decay_penalty_type
        n_layers: int = len(layer_sizes)
        n_bypass_layers: int = len(bypass_layer_sizes)

        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if not isinstance(bypass_weight_init_stddevs, SequenceCollection):
            bypass_weight_init_stddevs = [bypass_weight_init_stddevs
                                         ] * n_bypass_layers
        if not isinstance(bypass_bias_init_consts, SequenceCollection):
            bypass_bias_init_consts = [bypass_bias_init_consts
                                      ] * n_bypass_layers
        if not isinstance(bypass_dropouts, SequenceCollection):
            bypass_dropouts = [bypass_dropouts] * n_bypass_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers

        self.activation_fns: SequenceCollection[ActivationFn] = [
            self._get_activation_class(f) for f in activation_fns
        ]
        self.weight_init_stddevs: SequenceCollection[
            float] = weight_init_stddevs
        self.bias_init_consts: SequenceCollection[float] = bias_init_consts
        self.dropouts: SequenceCollection[float] = dropouts
        self.bypass_activation_fns: SequenceCollection[ActivationFn] = [
            self.activation_fns[0]
        ] * n_bypass_layers
        self.bypass_weight_init_stddevs: SequenceCollection[
            float] = bypass_weight_init_stddevs
        self.bypass_bias_init_consts: SequenceCollection[
            float] = bypass_bias_init_consts

        super(RobustMultitask, self).__init__()

        # Add shared layers
        self.shared_layers: nn.Sequential = self._build_layers(
            n_features, layer_sizes, self.activation_fns, dropouts,
            self.weight_init_stddevs, self.bias_init_consts)

        # Add task-specific bypass layers
        self.bypass_layers: nn.ModuleList = nn.ModuleList([
            self._build_layers(n_features, bypass_layer_sizes,
                               self.bypass_activation_fns, bypass_dropouts,
                               self.bypass_weight_init_stddevs,
                               self.bypass_bias_init_consts)
            for _ in range(n_tasks)
        ])

        # Output layers for each task
        self.output_layers: nn.ModuleList = nn.ModuleList([
            nn.Linear(layer_sizes[-1] + bypass_layer_sizes[-1], n_classes)
            for _ in range(n_tasks)
        ])

    def _build_layers(self, input_size, layer_sizes, activation_fns, dropouts,
                      weight_init_stddevs, bias_init_consts):
        """Helper function to build layers with activations and dropout"""

        prev_size = input_size
        layers = []

        for i, size in enumerate(layer_sizes):
            layer = nn.Linear(prev_size, size)
            try:
                nn.init.trunc_normal_(layer.weight, std=weight_init_stddevs[i])
            except IndexError:
                logger.warning(
                    "Warning: Wrong number of weight_init_stddevs specified. When passing weight_init_stddevs as a list, the length of the list should be equal to the number of layers."
                )
                logger.warning(
                    "Using default weight initialization: truncated normal with std=0.02"
                )
                nn.init.trunc_normal_(layer.weight, std=0.02)
            try:
                nn.init.constant_(layer.bias, bias_init_consts[i])
            except IndexError:
                logger.warning(
                    "Warning: Wrong number of bias_init_consts specified. When passing bias_init_consts as a list, the length of the list should be equal to the number of layers."
                )
                logger.warning(
                    "Using default bias initialization: constant=1.0")
                nn.init.constant_(layer.bias, 1.0)

            layers.append(layer)

            try:
                layers.append(activation_fns[i])
            except IndexError:
                logger.warning(
                    "Warning: Mismatch in number of activation functions and layers detected."
                )
                pass
            try:
                layers.append(nn.Dropout(dropouts[i]))
            except IndexError:
                logger.warning(
                    "Warning: Mismatch in number of dropouts and layers detected."
                )
                pass

            prev_size = size

        sequential = nn.Sequential(*layers)
        return sequential

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The Model output tensor of shape (batch_size, n_tasks, n_outputs).

        * When self.mode = `regression`,
            It consists of the output of each task.
        * When self.mode = `classification`,
            It consists of the probability of each class for each task.

        torch.Tensor, optional
            This is only returned when self.mode = `classification`, the output consists of the
            logits for classes before softmax.
        """
        task_outputs: List[torch.Tensor] = []

        # Shared layers
        shared_output = self.shared_layers(x)

        # Bypass layers
        for task in range(self.n_tasks):
            bypass_output = self.bypass_layers[task](x)

            # Concatenating outputs of shared layers and each task's bypass layers
            combined_output = torch.cat([shared_output, bypass_output], dim=1)

            # Task specific output layer
            task_output = self.output_layers[task](combined_output)
            task_outputs.append(task_output)

        output = torch.stack(task_outputs, dim=1)

        if self.mode == 'classification':
            if self.n_tasks == 1:
                logits = output.view(-1, self.n_classes)
                softmax_dim = 1
            else:
                logits = output.view(-1, self.n_tasks, self.n_classes)
                softmax_dim = 2
            proba = F.softmax(logits, dim=softmax_dim)
            return proba, logits
        else:
            return output.squeeze(-2)

    def regularization_loss(self):
        """Compute the regularization loss for the model."""
        reg_loss = 0.0

        for param in self.parameters():
            if self.weight_decay_penalty_type == "l1":
                reg_loss += torch.sum(torch.abs(param))
            elif self.weight_decay_penalty_type == "l2":
                reg_loss += torch.sum(torch.pow(param, 2))

        return self.weight_decay_penalty * reg_loss

    def _get_activation_class(self, activation_name: ActivationFn) -> Callable:
        """Get the activation class from the name of the activation function.

        Parameters
        ----------
        activation_name: str
            The name of the activation function.

        Returns
        -------
        Callable
            The activation function class.
        """
        if isinstance(activation_name, str):
            return getattr(nn, activation_name)
        elif isinstance(activation_name, nn.Module):
            return activation_name
        else:
            raise ValueError(
                f"Invalid activation function: {activation_name}. Only activations of type nn.Module"
            )
