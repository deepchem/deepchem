from torch import nn as nn

class ConvODEAutoEncoder(nn.Module):
  """
  Joins all the components of neural network together
  """

  def __init__(self, encoder: nn.Module, system_dynamics: nn.Module,
               decoder: nn.Module):
    """

    Parameters
    ----------
    encoder: nn.Module
      Encoder Neural Network Block
    decoder: nn.Module
      Decoder Neural Network Block
    system_dynamics: nn.Module
      Neural Network that learns system dynamics

    Returns
    -------
    torch.Tensor
    """

    super(ConvODEAutoEncoder, self).__init__()
    self.encoder = ConvODEEncoderLayer()
    self._system_dynamics = SystemDynamics()
    self._system_dynamics_odeblock = ODEBlock(
        system_dynamics=self._system_dynamics)
    self.decoder = ConvODEDecoderLayer()

  def forward(self, x):
    x = self.encoder(x)
    x = self._system_dynamics_ode_block(x)
    x = self.decoder(x)
    return x
<<<<<<< HEAD


class ConvODEAutoEncoderModel(TorchModel):

  def __init__(self):
    model = ConvODEAutoEncoder()
    loss = L2Loss()
    output_types = ['prediction']

    super(ConvODEAutoEncoderModel, self).__init__(
      model, loss= loss, output_types=output_types, **kwargs)

  def _prepare_batch(self, batch):
    pass
    
=======
>>>>>>> a7cc48e3 (moved ode layers to layers.py)
