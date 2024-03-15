import logging
import numpy as np
from collections.abc import Sequence as SequenceCollection
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')
from deepchem.metrics import to_one_hot
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.utils.typing import OneOrMany, ActivationFn
from deepchem.utils.pytorch_utils import get_activation
from typing import List, Tuple
from deepchem.models.losses import L2Loss
logger = logging.getLogger(__name__)


class RobustMultitaskModel(nn.Module):
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
                 layer_sizes: List[int] = [
                     1000,
                 ],
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 dropouts: float = 0.5,
                 activation_fns: str = "relu",
                 n_classes: int = 2,
                 bypass_layer_sizes: List[int] = [
                     100,
                 ],
                 bypass_weight_init_stddevs: List[float] = [.02],
                 bypass_bias_init_consts: List[float] = [1.],
                 bypass_dropouts: List[float] = [.5],
                 mode: str = "classification"):
        """  Create a RobustMultitaskClassifier.

        Parameters
        ----------
        n_tasks: int
            number of tasks
        n_features: int
            number of features
        layer_sizes: list
            the size of each dense layer in the network.  The length of this list determines the number of layers.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization of each layer.  The length
            of this list should equal len(layer_sizes).  Alternatively this may be a single value instead of a list,
            in which case the same value is used for every layer.
        bias_init_consts: list or loat
            the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        weight_decay_penalty: float
            the magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str
            the type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float
            the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        activation_fns: list or object
            the Tensorflow activation function to apply to each layer.  The length of this list should equal
            len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer.
        n_classes: int
            the number of classes
        bypass_layer_sizes: list
            the size of each dense layer in the bypass network. The length of this list determines the number of bypass layers.
        bypass_weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization of bypass layers.
            same requirements as weight_init_stddevs
        bypass_bias_init_consts: list or float
            the value to initialize the biases in bypass layers
            same requirements as bias_init_consts
        bypass_dropouts: list or float
            the dropout probablity to use for bypass layers.
            same requirements as dropouts
        mode: str
            Whether the model should perform classification or regression on the dataset
        """
        super(RobustMultitaskModel, self).__init__()
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")
        n_layers = len(layer_sizes)
        n_bypass_layers = len(bypass_layer_sizes)
        self.n_bypass_layers = n_bypass_layers
        self.n_layers = n_layers
        self.n_tasks = n_tasks
        self.n_features = n_features
        self.n_classes = n_classes
        self.mode = mode
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers

        if not isinstance(bypass_weight_init_stddevs, SequenceCollection):
            bypass_weight_init_stddevs = [bypass_weight_init_stddevs
                                         ] * n_bypass_layers
        if not isinstance(bypass_bias_init_consts, SequenceCollection):
            bypass_bias_init_consts = [bypass_bias_init_consts
                                      ] * n_bypass_layers
        if not isinstance(bypass_dropouts, SequenceCollection):
            bypass_dropouts = [bypass_dropouts] * n_bypass_layers
        
        self.activation_fns = [get_activation(i) for i in activation_fns]
        

        # Adding the shared represenation.
        list_layers: List[nn.Module] = []
        in_size: int = n_features

        for size, weight_stddev, bias_const, dropout, activation_fn in zip(
                layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
                self.activation_fns):
            layer = nn.Linear(in_size, size)
            nn.init.trunc_normal_(layer.weight, 0, weight_stddev)
            if layer.bias is not None:
                layer.bias = nn.Parameter(
                    torch.full(layer.bias.shape, bias_const))
            layer.weight_stddev = weight_stddev
            layer.bias_const = bias_const
            dropout_layer = nn.Dropout(dropout)
            layer_act = activation_fn
            list_layers.append(layer)
            list_layers.append(dropout_layer)
            
            in_size = size

        self.shared = nn.Sequential(*list_layers)

        # Adding Task specific layers.
        self.bypass_layers = nn.ModuleList()
        for task in range(self.n_tasks):
            task_layers = []
            in_size = n_features
            for bypass_size, bypass_weight_stddev, bypass_bias_const, bypass_dropout, bypass_activation_fn in zip(
                    bypass_layer_sizes, bypass_weight_init_stddevs,
                    bypass_bias_init_consts, bypass_dropouts,
                    self.activation_fns):
                layer_task = nn.Linear(in_size, bypass_size)
                nn.init.trunc_normal_(layer_task.weight, 0,
                                      bypass_weight_stddev)
                if layer_task.bias is not None:
                    layer_task.bias = nn.Parameter(
                        torch.full(layer_task.bias.shape,
                                   bypass_bias_const))
                layer_task.weight_stddev = bypass_weight_stddev
                layer_task.bias_const = bypass_bias_const
                dropout_layer_bypass = nn.Dropout(bypass_dropout)
                layer_act_bypass = bypass_activation_fn
                task_layers.append(layer_task)
                task_layers.append(dropout_layer_bypass)
                in_size = size
            task_layer = nn.Sequential(*task_layers)
        self.bypass_layers.append(task_layer)

        self.classifier = nn.LazyLinear(self.n_classes)
        self.regressor = nn.LazyLinear(1)

    def forward(self, X):
        X_bypass = torch.Tensor.new_tensor(X, requires_grad=True)
        shared_weights = self.shared(X)

        outputs_bypass = []
        for modules in self.bypass_layers:
            X_bypass = modules(X_bypass)
            outputs_bypass.append(X_bypass)
        out = []
        for i in outputs_bypass:
            output = torch.cat((shared_weights, i), dim=1)
            out.append(output)

        task_outputs = []
        logits = []
        if self.mode == "classification":
            for j in out:
                y = self.classifier(j)
                task_outputs.append(y)
            for output in task_outputs:
                softmax = nn.Softmax()
                logit = softmax(output)
                logits.append(logit)
            outs = [task_outputs, logits]
        if self.mode == "regression":
            for j in out:
                y = self.regressor(j)
                task_outputs.append(y)
            outs = task_outputs
        return outs


class RobustMultitask(TorchModel):
    def __init__(self,
                 n_tasks,
                 n_features,
                 layer_sizes=[1000,],
                 weight_init_stddevs=0.02,
                 bias_init_consts=1.0,
                 dropouts=0.5,
                 activation_fns="relu",
                 n_classes=2,
                 bypass_layer_sizes=[100,],
                 weight_decay_penalty="l1",
                 weight_decay_penalty_type=0.0,
                 bypass_weight_init_stddevs=[.02],
                 bypass_bias_init_consts=[1.],
                 bypass_dropouts=[.5],
                 mode="classification",
                 **kwargs):
        self.mode = mode
        self.n_classes = n_classes
        self.n_tasks = n_tasks

        self.model = RobustMultitaskModel(self,
                                          n_tasks=n_tasks,
                                          n_features=n_features,
                                          layer_sizes=layer_sizes,
                                          weight_init_stddevs=weight_init_stddevs,
                                          bias_init_consts=bias_init_consts,
                                          dropouts=dropouts,
                                          activation_fns=activation_fns,
                                          n_classes=n_classes,
                                          bypass_layer_sizes=bypass_layer_sizes,
                                          bypass_weight_init_stddevs=bypass_weight_init_stddevs,
                                          bypass_bias_init_consts=bypass_bias_init_consts,
                                          bypass_dropouts=bypass_dropouts,
                                          mode=mode)

        if weight_decay_penalty != 0:
            weights_shared = [layer.weight for layer in self.model.shared_layers]
            if weight_decay_penalty_type == 'l1':
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.abs(w).sum() for w in weights_shared]))
            else:
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.square(w).sum() for w in weights_shared]))
        else:
            regularization_loss = None

        loss = L2Loss()

        if self.mode == 'classification':
            output_types = ['prediction', 'loss']
        else:
            output_types = ["prediction"]

        super(RobustMultitask, self).__init__(self.model,
                                              loss=loss,
                                              output_types=output_types,
                                              regularization_loss=regularization_loss,
                                              **kwargs)

    def default_generator(self,
                            dataset,
                            epochs=1,
                            mode='fit',
                            deterministic=True,
                            pad_batches=True):

        for epoch in range(epochs):

            for (X_b, y_b, w_b,
                ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                            deterministic=deterministic,
                                            pad_batches=pad_batches):

                if self.mode == 'classification':
                    if y_b is not None:
                        y_b = to_one_hot(y_b.flatten(), self.n_classes)\
                            .reshape(-1, self.n_tasks, self.n_classes)

                dropout = np.array(0.) if mode == 'predict' else np.array(1.)

                yield ([X_b, dropout], [y_b], [w_b])
