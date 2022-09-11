import torch
import torch.nn as nn

import numpy as np
import torchdyn
from torchdyn.core import NeuralODE
from deepchem.models import TorchModel
from deepchem.models.losses import L2Loss
import tqdm

import torch.nn.functional as F
from deepchem.models.layers import LinearModule
from deepchem.utils.typing import ActivationFn, OneOrMany
from typing import Iterable, Optional, Callable, Union, Generator


class NeuralODEModel(TorchModel):
  """A General Neural ODE API to learn dynamics of the system. The Interface
  allows users to
  
  - Configurable the neural net modelling system dynamics in terms of number of layers,
  dropouts, weight initialisation
  - Configurable ode solver, allowing to set various sensitivity, adjoint methods for
  ode solvers
  - Custom Wrapper NN to NeuralODE enabling to use for deepchem tasks, and fit on deepchem 
  datasets
  
  TODO : Add Usage Example

  """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               vector_field_config: dict,
               layer_filters: OneOrMany[int],
               weight_init_stddevs: OneOrMany[float] = 0.02,
               bias_init_consts: OneOrMany[float] = 1.0,
               weight_decay_penalty: float = 0.0,
               weight_decay_penalty_type: str = 'l2',
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[ActivationFn] = 'relu',
               residual: bool = True,
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

    self.n_tasks, self.n_features, self.mode = n_tasks, n_features, mode

    # Linear Neural Network to model dynamics of system
    self.vector_field = LinearModule(
        n_features=vector_field_config['n_features'],
        layer_filters=vector_field_config['layer_filters'],
        dropouts=vector_field_config['dropouts'],
        activation_fns=vector_field_config['activation_fns'],
        weight_init_stddevs=vector_field_config['weight_init_stddevs'],
        bias_init_consts=vector_field_config['bias_init_consts'],
        residual=vector_field_config['residual'],
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

    self.model = LinearModule(n_features=n_features,
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

  # TODO : Write custom fit method
  def fit(self):
    pass
