import torch
import torch.nn as nn

import numpy as np
import torchdyn
from deepchem.models import TorchModel
from deepchem.models.losses import L2Loss
import tqdm

class NeuralODE(nn.Module):

  def __init__(self, 
               field: nn.Module = None, 
               n_tasks=1,
               solver='tsit', 
               order=1, 
               atol=0.001, 
               rtol=0.001,
               sensitivity='autograd', 
               atol_adjoint=0.0001, 
               rtol_adjoint=0.0001, 
               interpolator = None, 
               integral_loss=None, 
               seminorm=False, 
               return_t_eval=True, 
               optimizable_params={}):
    
    super().__init__()
    if field is not None:
      self.field = field
    else:
      self.field = nn.Sequential(
          nn.Linear(100,100))
    
    self.ode = torchdyn.core.NeuralODE(self.field, solver, order, atol, rtol, 
                  sensitivity, solver_adjoint, atol_adjoint, rtol_adjoint, 
                  interpolator, integral_loss, seminorm, return_t_eval, 
                  optimizable_params)
    
    self.input_layer = nn.LazyLinear(100)
    self.output_layer = nn.LazyLinear(n_tasks)

  def forward(self, inputs):
    x, dropout, t = inputs
    batch_size = x.shape[0]

    x = self.input_layer(x)
    x = self.ode(x, t)
    x = self.output_layer(x[0])
    
    return [x]


class NeuralODEModel(TorchModel):
  
  def __init__(self, 
               mode, 
               n_tasks, 
               n_features, 
               t_span=torch.Tensor([0., 1.]), 
               field:nn.Module = None, solver='tsit', 
               order=1, 
               atol=0.001, 
               rtol=0.001,
               sensitivity='autograd', 
               atol_adjoint=0.0001, 
               rtol_adjoint=0.0001, 
               interpolator = None, 
               integral_loss=None, 
               seminorm=False, 
               return_t_eval=True, 
               optimizable_params={}) -> None:

    self.n_tasks, self.n_features, self.mode = n_tasks, n_features, mode
    self.model = NeuralODE(field, solver, order, atol, rtol, sensitivity, 
                    atol_adjoint, rtol_adjoint, interpolator, integral_loss, 
                    seminorm, return_t_eval, optimizable_params)
    
    self.t_span = t_span

    loss = L2Loss()
    output_types = ['prediction']

    super(NeuralODEModel, self).__init__(self.model, loss=loss, output_types=output_types, **kwargs)
    
  def default_generator(self, dataset, epochs=1, mode='fit', deterministic=True, pad_batches=True):
    
    for epoch in tqdm(range(epochs)):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if mode == 'predict':
          dropout = np.array(0.0)
        else:
          dropout = np.array(1.0)
        yield ([X_b, dropout, self.t_span], [y_b], [w_b])
