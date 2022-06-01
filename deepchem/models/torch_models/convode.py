import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
import deepchem.models.torch_model import TorchModel
from deepchem.models.losses import L2Loss


class Encoder(nn.Module):
  
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(3,16,3,2)
    self.conv2 = nn.Conv2d(16,32,3,2)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))

    return out



class f(nn.Module):
  def __init__(self, dim=0):
    super(f, self).__init__()
    
    self.conv1 = nn.Conv2d(32,48,7)
    self.conv2 = nn.Conv2d(48,64,5)
    self.conv3 = nn.Conv2d(64,32,5)
    self.conv4 = nn.Conv2d(32,16,5)

    self.layer = nn.Sequential(
        nn.ConvTranspose2d(16,32,5),
        nn.ConvTranspose2d(32,64,5),
        nn.ConvTranspose2d(64,48,5),
        nn.ConvTranspose2d(48,32,7)
    )

  def forward(self, t, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = F.dropout(out)
    out = self.conv4(out)    
    out = self.layer(out)
    
    return out


class ODEBlock(nn.Module):

  def __init__(self, f):
    super(ODEBlock, self).__init__()
    self.f = f
    self.int_time = torch.Tensor([0,1]).float()

  def forward(self,x):
    self.int_time = self.int_time.type_as(x)
    out = torchdiffeq.odeint_adjoint(self.f, x, self.int_time)
    #print(f"out[1].shape {out[1].shape}")
    return out[1]



class Decoder(nn.Module):

  def __init__(self):
    super(Decoder, self).__init__()
    self.layer = nn.Sequential(
        nn.ConvTranspose2d(32,16,3,2),
        nn.ConvTranspose2d(16,3,3,2),
    )
  
  def forward(self, x):
    return self.layer(x)


class ConvODEAutoEncoder(nn.Module):
  
  def __init__(self):
    super(ConvODEAutoEncoder, self).__init__()
    self.encoder = Encoder()
    self.dynamics_block = ODEBlock(f())
    self.decoder = Decoder()

  def forward(self, x):
    x = self.encoder(x)
    #print(f"X.SHAPE {x.shape}")
    x = self.dynamics_block(x)
    #print(f"X.SHAPE {x.shape}")
    x = self.decoder(x)
    #print(f"X.SHAPE {x.shape}")
    return x


class ConvODEAutoEncoderModel(TorchModel):

  def __init__(self):
    model = ConvODEAutoEncoder()
    loss = L2Loss()
    output_types = ['prediction']

    super(ConvODEAutoEncoderModel, self).__init__(
      model, loss= loss, output_types=output_types, **kwargs)

  def _prepare_batch(self, batch):
    pass
    