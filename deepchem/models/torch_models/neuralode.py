import torch
import torch.nn as nn

import numpy as np
from torchdyn.core import NeuralODE
from deepchem.models import TorchModel
from deepchem.models.losses import L2Loss
import tqdm

import torch.nn.functional as F
from deepchem.utils.typing import ActivationFn, OneOrMany
from deepchem.utils.pytorch_utils import get_activation
from typing import Iterable, Optional, Callable, Union, Generator
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection


class NeuralODEModel(TorchModel):
  """A General Neural ODE API to learn dynamics of the system. The Interface
  allows users to
  
  - Configurable the neural net modelling system dynamics in terms of number of layers,
  dropouts, weight initialisation
  - Configurable ode solver, allowing to set various sensitivity, adjoint methods for
  ode solvers
  - Custom Wrapper NN to NeuralODE enabling to use for deepchem tasks, and fit on deepchem 
  datasets
  
  Examples
  --------
  >>> import deepchem as dc
  >>> model = NeuralODEModel()
  >>> X = np.random.rand(n_samples, 10, n_features)
  >>> y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>>
  >>> regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
  >>> model.fit(dataset, nb_epoch=1)
  >>> scores = model.evaluate(dataset, [regression_metric])

  """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               layer_filters: OneOrMany[int],
               ode_field_layer_filters: OneOrMany[int],
               weight_init_stddevs: OneOrMany[float] = 0.02,
               ode_field_weight_init_stddevs: OneOrMany[float] = 0.02,
               bias_init_consts: OneOrMany[float] = 1.0,
               ode_field_bias_init_consts: OneOrMany[float] = 1.0,
               weight_decay_penalty: float = 0.0,
               ode_field_weight_decay_penalty: float = 0.0,
               weight_decay_penalty_type: str = 'l2',
               ode_field_weight_decay_penalty_type: str = 'l2',
               dropouts: OneOrMany[float] = 0.5,
               ode_field_dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[ActivationFn] = 'relu',
               ode_field_activation_fns: OneOrMany[ActivationFn] = 'relu',
               residual: bool = True,
               ode_field_residual: bool = True,
               mode: str = 'regression',
               t_span=torch.Tensor([0., 1.]),
               solver: str = 'tsit5',
               order: int = 1,
               atol: float = 0.001,
               rtol: float = 0.001,
               sensitivity: str = 'autograd',
               solver_adjoint: Union[str, nn.Module] = None,
               atol_adjoint: float = 0.0001,
               rtol_adjoint: float = 0.0001,
               interpolator=None,
               integral_loss=None,
               seminorm: bool = False,
               return_t_eval: bool = True,
               optimizable_params: Union[Iterable, Generator] = {},
               **kwargs) -> None:
    """NeuralODE TorchModel, with custom fit methods for 

      TODO : Flexibility to use Convolutional/ Linear NN as modelling network

      Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: int
      number of features
    layer_filters: list
      the number of output filters for each linear layer in the network.
      The length of this list determines the number of layers.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization
      of each layer.  The length of this list should equal len(layer_filters)+1,
      where the final element corresponds to the dense layer.  Alternatively this
      may be a single value instead of a list, in which case the same value is used
      for every layer.
    bias_init_consts: list or float
      the value to initialize the biases in each layer to.  The length of this
      list should equal len(layer_filters)+1, where the final element corresponds
      to the dense layer.  Alternatively this may be a single value instead of a
      list, in which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float
      the dropout probability to use for each layer.  The length of this list should equal len(layer_filters).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer
    activation_fns: str or list
      the torch activation function to apply to each layer. The length of this list should equal
      len(layer_filters).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer, 'relu' by default
    mode: str
      Either 'classification' or 'regression'
    residual: bool
      if True, the model will be composed of pre-activation residual blocks instead
      of a simple stack of linear layers.
    t_span: torch.Tensor
      range to perform integration over
    ode_block: nn.Module
      whether to use neural ode
    vector_field: nn.Module 
      the vector field, called with vector_field(t, x) for vector_field(x). In the second case, the Callable is automatically wrapped for consistency
    solver: str
      the ode solver to use, visit https://torchdyn.readthedocs.io/en/latest/source/torchdyn.numerics.html#module-torchdyn.numerics.solvers for information on this
    order: int
      Order of the ODE. Defaults to 1.
    atol: float
      Absolute tolerance of the solver. Defaults to 1e-4
    rtol: float
      Relative tolerance of the solver. Defaults to 1e-4.
    sensitivity: str
      Sensitivity method [‘autograd’, ‘adjoint’, ‘interpolated_adjoint’]. Defaults to ‘autograd’.
    solver_adjoint: Union[Callable, None],
      ODE solver for the adjoint. Defaults to None.
    atol_adjoint: float
      Defaults to 1e-6
    rtol_ajoint: float
      Defaults to 1e-6
    integral_loss: Callable
      Defaults to None
    seminorm: bool
      Whether to use seminorms for adaptive stepping in backsolve adjoints. Defaults to False.
    return_t_eval: bool
      Whether to return (t_eval, sol) or only sol. Useful for chaining NeuralODEs in nn.Sequential.
    optimizable_parameters: Union[Iterable, Generator]
      parameters to calculate sensitivies for. Defaults to ()

    """

    self.n_tasks = n_tasks
    self.n_features = n_features
    self.mode = mode

    class _LinearFieldImpl(nn.Module):
        """A Flexible Linear FeedForward Network Module. The interface allows users to
        configure the neural net in terms of number of layers, dropouts, weight initialisation
        """

        def __init__(self,
                     layer_filters: OneOrMany[int],
                     n_features: int = 1,
                     dropouts: OneOrMany[float] = 0.5,
                     activation_fns: OneOrMany[ActivationFn] = 'relu',
                     weight_init_stddevs: OneOrMany[float] = 0.02,
                     bias_init_consts: OneOrMany[float] = 1.0,
                     residual: bool = True,
                     ode_block: nn.Module = None,
                     enable_ffn: bool = True,
                     mode: str = 'regression'):

            self.n_tasks = n_tasks
            self.dropouts = dropouts
            self.ode_block = ode_block
            self.residual = residual
            self.enable_ffn = enable_ffn
            self.mode = mode

            n_layers = len(layer_filters)

            if len(layer_filters) == 1:
                layer_filters = layer_filters * 2
            if not isinstance(dropouts, SequenceCollection):
                dropouts = [dropouts] * n_layers
            if isinstance(activation_fns,
                          str) or not isinstance(activation_fns, SequenceCollection):
                activation_fns = [activation_fns] * n_layers
            if not isinstance(weight_init_stddevs, SequenceCollection):
                weight_init_stddevs = [weight_init_stddevs] * n_layers
            if not isinstance(bias_init_consts, SequenceCollection):
                bias_init_consts = [bias_init_consts] * n_layers

            self.activation_fns = [get_activation(f) for f in activation_fns]
            self.dropouts = dropouts

            self.net = nn.ModuleList()

            in_shape = n_features

            for out_shape, weight_stddev, bias_const in zip(layer_filters,
                                                            weight_init_stddevs,
                                                            bias_init_consts):

                layer = nn.Linear(in_features=in_shape, out_features=out_shape, bias=True)

                nn.init.normal_(layer.weight, 0, weight_stddev)

                if layer.bias is not None:
                    layer.bias = nn.Parameter(torch.full(layer.bias.shape, bias_const))

                self.layer.append(layer)

            self.classifier_ffn = nn.LazyLinear(self.n_tasks * self.n_classes)
            self.output_layer = nn.LazyLinear(self.n_tasks)

            super(_LinearFieldImpl, self).__init__()

        def forward(self, inputs):
            """
            Parameters
            ----------
            inputs: list of torch.Tensor
              Input Tensors
            Returns
            -------
            torch.Tensor
              Output Tensor
            """

            if isinstance(inputs, torch.Tensor):
                x, dropout_switch = inputs, None
            else:
                x, dropout_switch = inputs

            prev_layer = x

            for layer, activation_fn, dropout in zip(self.layers, self.activation_fns,
                                                     self.dropouts):
                x = layer(x)

                if dropout > 0. and dropout_switch:
                    x = F.dropout(x, dropout)

                    if self.residual and x.shape[1] == prev_layer.shape[1]:
                        x = x + prev_layer

                    if activation_fn is not None:
                        x = self.activation_fn(x)

                    prev_layer = x

            if self.ode_block is not None:
                x, _ = self.ode_block(x)

            outputs = x

            if self.enable_ffn:
                batch_size = x.shape[0]

                if self.mode == "classification":

                    logits = self.classifier_ffn(x)
                    logits = logits.view(batch_size, self.n_tasks, self.n_classes)
                    output = F.softmax(logits, dim=2)
                    outputs = [output, logits]

                else:
                    output = self.output_layer(x)
                    output = output.view(batch_size, self.n_tasks)

                    outputs = [output]

            return outputs

    # Linear Neural Network to model dynamics of system
    self.vector_field = _LinearFieldImpl(
        n_features=n_features,
        layer_filters=ode_field_layer_filters,
        dropouts=ode_field_dropouts,
        activation_fns=ode_field_activation_fns,
        weight_init_stddevs=ode_field_weight_init_stddevs,
        bias_init_consts=ode_field_bias_init_consts,
        residual=ode_field_residual,
        ode_block=None,
        enable_ffn=False)

    self.neural_ode = NeuralODE(vector_field=self.vector_field,
                                solver=solver,
                                order=order,
                                atol=atol,
                                rtol=rtol,
                                sensitivity=sensitivity,
                                solver_adjoint=solver_adjoint,
                                atol_adjoint=atol_adjoint,
                                rtol_adjoint=rtol_adjoint,
                                interpolator=interpolator,
                                integral_loss=integral_loss,
                                seminorm=seminorm,
                                return_t_eval=return_t_eval,
                                optimizable_params=optimizable_params)

    self.model = _LinearFieldImpl(n_features=n_features,
                                  layer_filters=layer_filters,
                                  dropouts=dropouts,
                                  activation_fns=activation_fns,
                                  weight_init_stddevs=weight_init_stddevs,
                                  bias_init_consts=bias_init_consts,
                                  residual=residual,
                                  ode_block=self.neural_ode,
                                  enable_ffn=True)

    self.t_span = t_span

    regularization_loss: Optional[Callable]

    if weight_decay_penalty != 0:
      weights = [layer.weight for layer in self.model.layers]
      if weight_decay_penalty_type == 'l1':
        regularization_loss = lambda: weight_decay_penalty * torch.sum(
            torch.stack([torch.abs(w).sum() for w in weights]))
      else:
        regularization_loss = lambda: weight_decay_penalty * torch.sum(
            torch.stack([torch.square(w).sum() for w in weights]))
    else:
      regularization_loss = None

    loss = L2Loss()

    output_types = ['prediction']

    super(NeuralODEModel,
          self).__init__(self.model,
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

    for epoch in tqdm(range(epochs)):
      for (t_b, Y_b, w_b,
           ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                         deterministic=deterministic,
                                         pad_batches=pad_batches):
        if mode == 'predict':
          dropout = np.array(0.0)
        else:
          dropout = np.array(1.0)
        yield ([t_b, dropout], Y_b, [w_b])
