import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from deepchem.models.torch_models.layers import Highway
import numpy as np

def test_highway():
  """Test invoking Highway."""
  width = 5
  batch_size = 2
  input = torch.rand(batch_size, width, dtype=torch.float32)
  layer = Highway(layer_shape=[width, 2])
  result = layer(input)
  assert result.shape == (batch_size, width)
  assert len(layer.parameters()) == 4

  # Creating a second layer should produce different results, since it has
  # different random weights.

  layer2 = Highway(layer_shape=[width, 2])
  result2 = layer2(input)

  assert not torch.allclose(result, result2)

  # But evaluating the first layer again should produce the same result as before.

  result3 = layer(input)
  assert torch.allclose(result, result3)


def test_highway_layer_shape():
    from deepchem.models.torch_models.layers import Highway
    width = 5
    batch_size = 10
    torch.manual_seed(21)
    inputs = torch.tensor([[0.4041, 0.3648, 0.5749, 0.0727, 0.7612],
                                   [0.8641, 0.5746, 0.3337, 0.4103, 0.2618],
                                   [0.0535, 0.5125, 0.2130, 0.2297, 0.2028],
                                   [0.0251, 0.5023, 0.0383, 0.0829, 0.4311],
                                   [0.0590, 0.3707, 0.7875, 0.5276, 0.3526],
                                   [0.2961, 0.7429, 0.2964, 0.9543, 0.9686],
                                   [0.8921, 0.7150, 0.8407, 0.8312, 0.9262],
                                   [0.3936, 0.1754, 0.1488, 0.7005, 0.4626],
                                   [0.9021, 0.3961, 0.2994, 0.1206, 0.4926],
                                   [0.0156, 0.0346, 0.1109, 0.6938, 0.4572]])

    layer = Highway(batch_size,width)

    tf_weights = np.load(
         'assets/highway_weights.npy',
         allow_pickle=True).item()
         
    with torch.no_grad():
        layer.H.weight.data.weight.data = torch.from_numpy(
         np.transpose(tf_weights['assets/TF_weights_H.npy']))
        layer.H.weight.data.bias.data = torch.from_numpy(
         tf_weights['assets/TF_bias_T.npy'])
        layer.T.weight.data.weight.data = torch.from_numpy(
         np.transpose(tf_weights['assets/TF_weights_T.npy']))
        layer.T.weight.data.bias.data = torch.from_numpy(
         tf_weights['assets/TF_bias_T.npy'])   
         
    output_tensor = torch.from_numpy(
         np.load('assets/highway_output.npy').astype(
             np.float32))
  
    assert torch.allclose(output, output_tensor, atol=1e-04)
    print(output)
    assert 1==2
    





