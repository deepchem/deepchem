import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from deepchem.models.torch_models.layers import Highway
import numpy as np

class Highway(torch.nn.Module):
  def __init__(self, layer_shape):
        super(Highway, self).__init__()

  def test_highway_layer1():
    
    width = 5
    batch_size = 2
    torch.manual_seed(42)

    input = torch.rand(batch_size, width, dtype=torch.float32)
    layer = Highway(layer_shape=[width, 2])
    result = layer(input)
    assert result.shape == (batch_size, width)
    assert len(list(layer.parameters())) == 0 

    result2 = layer(input)
    assert torch.allclose(result, result2)

  def test_highway_layer2():
   
    width = 5
    batch_size = 2
    torch.manual_seed(42)

    input = torch.rand(batch_size, width, dtype=torch.float32)
    layer = Highway(layer_shape=[width, 2])
    result = layer(input)

    layer2 = Highway(layer_shape=[width, 2])
    result2 = layer2(input)

    assert not torch.allclose(result, result2)

def test_highway_layer_shape():
    from deepchem.models.torch_models.layers import Highway
    width = 5
    batch_size = 2
    torch.manual_seed(21)
    layer_shape = [batch_size,width]
    inputs = torch.tensor([[-0.13437884, 0.19134927, -0.42180598,  1.5959879 ,  0.04390939],
                           [ 1.1014341,  0.33973673, -0.34050095,  0.03709555,  0.5169554 ]])

    layer = Highway(layer_shape=layer_shape)

    tf_weights = np.load(
         'assets/highway_weights.npy',
         allow_pickle=True).item()

    with torch.no_grad():
        layer.linear_H.weight.data = torch.from_numpy(
         np.transpose(tf_weights['dense_H/kernel']))
        
        layer.linear_T.weight.data = torch.from_numpy(
         np.transpose(tf_weights['dense_T/kernel']))

    output = layer(inputs)
    output_tensor = torch.from_numpy(
         np.load('assets/highway_output.npy').astype(
             np.float32))
    assert torch.allclose(output, output_tensor, atol=1e-04)
