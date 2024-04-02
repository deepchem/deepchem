import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchem.data import Dataset
from deepchem.models import losses
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.utils.pytorch_utils import get_activation
from deepchem.utils.typing import OneOrMany, ActivationFn, LossFn

import logging
from collections.abc import Sequence as SequenceCollection
from typing import List, Tuple, Callable, Literal, Optional, Union

logger = logging.getLogger(__name__)


class ProgressiveMultitask(nn.Module):
    """Implements a progressive multitask neural network in PyTorch.

    Progressive networks allow for multitask learning where each task
    gets a new column of weights and lateral connections to previous tasks
    are added to the network. As a result, there is no exponential
    forgetting where previous tasks are ignored.

    Examples
    --------
    >>> import torch
    >>> import deepchem as dc
    >>> n_tasks = 4
    >>> n_features = 1024
    >>> n_classes = 2
    >>> sample = torch.randn(16, n_features)
    >>> model = dc.models.torch_models.ProgressiveMultitask(n_tasks=n_tasks, n_features=n_features, layer_sizes=[1024, 1024], mode="classification", n_classes=n_classes)
    >>> output = model(sample)
    >>> print(output[0].type(), output[1].type())
    torch.FloatTensor torch.FloatTensor
    >>> print(output[0].shape, output[1].shape)
    torch.Size([16, 4, 2]) torch.Size([16, 4, 2])

    References
    ----------
    See [1]_ for a full description of the progressive architecture

    .. [1] Rusu, Andrei A., et al. "Progressive neural networks." arXiv preprint
        arXiv:1606.04671 (2016).
    """

    def __init__(self,
                 n_tasks: int,
                 n_features: int,
                 layer_sizes: List[int] = [1000],
                 mode: Literal['regression', 'classification'] = 'regression',
                 alpha_init_stddevs: OneOrMany[float] = 0.02,
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 weight_decay_penalty: float = 0.0,
                 weight_decay_penalty_type: str = "l2",
                 activation_fns: OneOrMany[ActivationFn] = 'relu',
                 dropouts: OneOrMany[float] = 0.5,
                 n_classes: int = 1):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        n_features: int
            Size of input feature vector.
        layer_sizes: list of ints
            List of layer sizes.
        mode: str
            Type of model.  Must be 'regression' or 'classification'.
        alpha_init_stddevs: float or list of floats
            Standard deviation for truncated normal distribution to initialize
            alpha parameters.
        weight_init_stddevs: float or list of floats
            Standard deviation for truncated normal distribution to initialize
            weight parameters.
        bias_init_consts: float or list of floats
            Constant value to initialize bias parameters.
        weight_decay_penalty: float
            Amount of weight decay penalty to use.
        weight_decay_penalty_type: str
            Type of weight decay penalty.  Must be 'l1' or 'l2'.
        activation_fns: str or list of str
            Name of activation function(s) to use.
        dropouts: float or list of floats
            Dropout probability.
        n_classes: int
            The number of classes to predict per task.
        """
        if weight_decay_penalty != 0.0:
            raise ValueError("Weight decay is not currently supported")

        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(alpha_init_stddevs, SequenceCollection):
            alpha_init_stddevs = [alpha_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers

        self.mode = mode
        self.n_tasks: int = n_tasks
        self.n_features: int = n_features
        self.layer_sizes: List[int] = layer_sizes
        self.n_classes: int = n_classes
        self.weight_init_stddevs: SequenceCollection[
            float] = weight_init_stddevs
        self.alpha_init_stddevs: SequenceCollection[float] = alpha_init_stddevs
        self.bias_init_consts: SequenceCollection[float] = bias_init_consts
        self.dropouts: SequenceCollection[float] = dropouts
        self.activation_fns: SequenceCollection[Callable] = [
            get_activation(f) for f in activation_fns
        ]

        super(ProgressiveMultitask, self).__init__()

        self.layers: nn.ModuleList = nn.ModuleList()
        self.adapters: nn.ModuleList = nn.ModuleList()
        self.alphas: nn.ModuleList = nn.ModuleList()

        for task in range(n_tasks):
            layer_list = []
            adapter_list = []
            alpha_list = []
            prev_size = n_features
            for i, size in enumerate(self.layer_sizes):
                layer_list.append(self._init_linear(prev_size, size, i))

                if task > 0:
                    if i > 0:
                        adapter, alpha = self._get_adapter(
                            task, prev_size, size, i)
                        adapter_list.append(adapter)
                        alpha_list.append(alpha)

                prev_size = size

            layer_list.append(
                self._init_linear(prev_size, self.n_classes,
                                  len(self.layer_sizes)))
            self.layers.append(nn.ModuleList(layer_list))
            if task > 0:
                adapter, alpha = self._get_adapter(task, prev_size,
                                                   self.n_classes,
                                                   len(self.layer_sizes))
                adapter_list.append(adapter)
                alpha_list.append(alpha)

                self.adapters.append(nn.ModuleList(adapter_list))
                self.alphas.append(nn.ParameterList(alpha_list))

    def _get_adapter(self, task: int, prev_size: int, size: int,
                     layer_num: int) -> Tuple[nn.ModuleList, torch.Tensor]:
        """Creates the adapter layer between previous tasks and the current layer.

        Parameters
        ----------
        task: int
            Task number.
        prev_size: int
            Size of previous layer.
        size: int
            Size of current layer.
        layer_num: int
            Layer number.

        Returns
        -------
        adapter: nn.Sequential
            Adapter layer.
        alpha: torch.Tensor
            Alpha parameter.
        """
        adapter = nn.ModuleList([
            self._init_linear(prev_size * task, prev_size, layer_num),
            self._init_linear(prev_size, size, layer_num, use_bias=False),
        ])
        alpha_init_stddev = (self.alpha_init_stddevs[layer_num]
                             if layer_num < len(self.layer_sizes) else
                             self.alpha_init_stddevs[-1])
        alpha = torch.empty(1, requires_grad=True)
        nn.init.trunc_normal_(alpha, std=alpha_init_stddev)
        return adapter, alpha

    def _init_linear(self,
                     in_features: int,
                     out_features: int,
                     layer_num: int,
                     use_bias: bool = True) -> nn.Linear:
        """Initialises nn.Linear layer weight and bias parameters.

        Parameters
        ----------
        in_features: int
            Size of input feature vector.
        out_features: int
            Size of output feature vector.
        layer_num: int
            Layer number.
        use_bias: bool
            Whether to use bias for Linear layer. Default to True.

        Returns
        -------
        layer: nn.Linear
            Linear layer with initialised parameters.
        """
        if layer_num < len(self.layer_sizes):
            weight_init_stddev = self.weight_init_stddevs[layer_num]
            bias_init_const = self.bias_init_consts[layer_num]
        elif layer_num == len(self.layer_sizes):
            weight_init_stddev = self.weight_init_stddevs[-1]
            bias_init_const = self.bias_init_consts[-1]

        layer = nn.Linear(in_features, out_features, bias=use_bias)
        nn.init.trunc_normal_(layer.weight, std=weight_init_stddev)

        if use_bias:
            nn.init.constant_(layer.bias, bias_init_const)

        return layer

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
        layer_logits: List[List[torch.Tensor]] = []
        for task in range(self.n_tasks):
            x_ = x
            layer_outputs = []
            for i, (layer, activation_fn, dropout) in enumerate(
                    zip(self.layers[task], self.activation_fns, self.dropouts)):
                x_ = layer(x_)

                if task > 0 and i > 0:
                    adapter_out = self._get_lateral_contrib(
                        layer_logits, task, i)
                    x_ = x_ + adapter_out

                if i < len(self.layer_sizes):
                    x_ = activation_fn(x_)
                    if dropout > 0.0:
                        x_ = F.dropout(x_, p=dropout, training=self.training)

                layer_outputs.append(x_)

            out_layer_idx = len(self.layer_sizes)
            x_ = self.layers[task][out_layer_idx](x_)
            if task > 0:
                adapter_out = self._get_lateral_contrib(layer_logits, task,
                                                        out_layer_idx)
                x_ = x_ + adapter_out

            layer_logits.append(layer_outputs)
            task_outputs.append(x_)

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
            return output

    def _get_lateral_contrib(self, logits, task, layer):
        adapter = self.adapters[task - 1][layer - 1]
        alpha = self.alphas[task - 1][layer - 1]
        prev_logits = [logits[t][layer - 1] for t in range(task)]
        adapter_input = torch.cat(prev_logits, dim=-1)
        adapter_input = alpha * adapter_input

        adapter_out = adapter[0](adapter_input)
        adapter_out = self.activation_fns[layer - 1](adapter_out)
        adapter_out = adapter[1](adapter_out)
        return adapter_out


class ProgressiveMultitaskModel(TorchModel):
    """Implements a progressive multitask neural network in PyTorch.

    Progressive networks allow for multitask learning where each task
    gets a new column of weights and lateral connections to previous tasks
    are added to the network. As a result, there is no exponential
    forgetting where previous tasks are ignored.

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import ProgressiveMultitaskModel
    >>> featurizer = dc.feat.CircularFingerprint(size=1024, radius=4)
    >>> tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer)
    >>> train_dataset, valid_dataset, test_dataset = datasets
    >>> n_tasks = len(tasks)
    >>> model = ProgressiveMultitaskModel(n_tasks, 1024, layer_sizes=[1024], mode='classification')
    >>> model.fit(train_dataset, nb_epoch=10)

    References
    ----------
    See [1]_ for a full description of the progressive architecture

    .. [1] Rusu, Andrei A., et al. "Progressive neural networks." arXiv preprint
        arXiv:1606.04671 (2016).
    """

    def __init__(self,
                 n_tasks: int,
                 n_features: int,
                 layer_sizes: List[int] = [1000],
                 mode: Literal['regression', 'classification'] = 'regression',
                 alpha_init_stddevs: OneOrMany[float] = 0.02,
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 weight_decay_penalty: float = 0.0,
                 weight_decay_penalty_type: str = "l2",
                 activation_fns: OneOrMany[ActivationFn] = 'relu',
                 dropouts: OneOrMany[float] = 0.5,
                 n_classes: Optional[int] = None,
                 n_outputs: Optional[int] = None,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        n_features: int
            Size of input feature vector.
        layer_sizes: list of ints
            List of layer sizes.
        mode: str
            Type of model.  Must be 'regression' or 'classification'.
        alpha_init_stddevs: float or list of floats
            Standard deviation for truncated normal distribution to initialize
            alpha parameters.
        weight_init_stddevs: float or list of floats
            Standard deviation for truncated normal distribution to initialize
            weight parameters.
        bias_init_consts: float or list of floats
            Constant value to initialize bias parameters.
        weight_decay_penalty: float
            Amount of weight decay penalty to use.
        weight_decay_penalty_type: str
            Type of weight decay penalty.  Must be 'l1' or 'l2'.
        activation_fns: str or list of str
            Name of activation function(s) to use.
        dropouts: float or list of floats
            Dropout probability.
        n_classes: int
            The number of classes to predict per task. Default to 2 for classification and 1 for regression.
        n_outputs: int
            The number of outputs to predict per task. Deprecated, use n_classes instead.
        """

        if n_outputs is not None:
            logger.warning(
                "n_outputs is deprecated and will be removed in future versions. Use n_classes instead."
            )
            if n_classes is not None and n_classes != n_outputs:
                raise ValueError(
                    "n_classes and n_outputs should have the same value if both are specified."
                )
            n_classes = n_outputs

        loss: losses.Loss
        if mode == 'regression':
            loss = losses.L2Loss()
            output_types = ['prediction']
            if n_classes is None:
                n_classes = 1
        elif mode == 'classification':
            loss = losses.SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
            if n_classes is None:
                n_classes = 2
        else:
            raise ValueError(f'Invalid mode: {mode}')

        model = ProgressiveMultitask(
            n_tasks=n_tasks,
            n_features=n_features,
            layer_sizes=layer_sizes,
            mode=mode,
            alpha_init_stddevs=alpha_init_stddevs,
            weight_init_stddevs=weight_init_stddevs,
            bias_init_consts=bias_init_consts,
            weight_decay_penalty=weight_decay_penalty,
            weight_decay_penalty_type=weight_decay_penalty_type,
            activation_fns=activation_fns,
            dropouts=dropouts,
            n_classes=n_classes)

        super(ProgressiveMultitaskModel,
              self).__init__(model, loss, output_types=output_types, **kwargs)

    def fit(self,
            dataset: Dataset,
            nb_epoch: int = 10,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 1000,
            deterministic: bool = False,
            restore: bool = False,
            variables: Optional[List[torch.nn.Parameter]] = None,
            loss: Optional[LossFn] = None,
            callbacks: Union[Callable, List[Callable]] = [],
            all_losses: Optional[List[float]] = None):

        for task in range(self.model.n_tasks):
            self.fit_task(dataset=dataset,
                          task=task,
                          nb_epoch=nb_epoch,
                          max_checkpoints_to_keep=max_checkpoints_to_keep,
                          checkpoint_interval=checkpoint_interval,
                          deterministic=deterministic,
                          restore=restore,
                          variables=variables,
                          loss=loss,
                          callbacks=callbacks,
                          all_losses=all_losses)

    def fit_task(self,
                 dataset: Dataset,
                 task: int,
                 nb_epoch: int = 10,
                 max_checkpoints_to_keep: int = 5,
                 checkpoint_interval: int = 1000,
                 deterministic: bool = False,
                 restore: bool = False,
                 variables: Optional[List[torch.nn.Parameter]] = None,
                 loss: Optional[LossFn] = None,
                 callbacks: Union[Callable, List[Callable]] = [],
                 all_losses: Optional[List[float]] = None):
        """Train this model on one task. Called by fit() to train each task sequentially.
        Calls fit_generator() internally.

        Parameters
        ----------
        dataset: Dataset
            the Dataset to train on
        task: int
            the task to train on
        nb_epoch: int
            the number of epochs to train for
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        deterministic: bool
            if True, the samples are processed in order.  If False, a different random
            order is used for each epoch.
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

        generator = self.default_generator(dataset,
                                           epochs=nb_epoch,
                                           deterministic=deterministic)

        variables = list(self.model.layers[task].parameters())

        if task > 0:
            variables += list(self.model.adapters[task - 1].parameters())
            variables += list(self.model.alphas[task - 1].parameters())

        self.fit_generator(generator=generator,
                           max_checkpoints_to_keep=max_checkpoints_to_keep,
                           checkpoint_interval=checkpoint_interval,
                           restore=restore,
                           variables=variables,
                           loss=loss,
                           callbacks=callbacks,
                           all_losses=all_losses)
