import torch
import torch.nn as nn
import torch.nn.init as init

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
  from deepchem.models.torch_models.layers import HighwayLayer
  width = 5
  batch_size = 10

  layer = HighwayLayer(inputs=inputs)
  layer_H = torch.randn([input_shape, out_channels])
  output = layer(inputs)
  output_tf = (5, 3)
     # Testing Shapes with TF Model Output
  assert output.shape == output_tf
  assert dense_H.shape == dense_H
  assert dense_T.shape == dense_T

     # Testing Shapes
  assert output.shape == (width,batch_size)
  assert layer.activation == torch.ReLU

