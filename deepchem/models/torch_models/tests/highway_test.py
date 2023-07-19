import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
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

    layer = HighwayLayer(inputs=inputs)

    tf_weights = np.load(
         'deepchem/models/tests/assets/Highway_weights.npy',
         allow_pickle=True).item()
    with torch.no_grad():
        layer.H.weight.data.weight.data = torch.from_numpy(
         np.transpose(tf_weights['deepchem/models/tests/assets/TF_weights_H.npy']))
        layer.H.weight.data.bias.data = torch.from_numpy(
         tf_weights['deepchem/models/tests/assets/TF_bias_T.npy'])
        layer.T.weight.data.weight.data = torch.from_numpy(
         np.transpose(tf_weights['deepchem/models/tests/assets/TF_weights_T.npy']))
        layer.T.weight.data.bias.data = torch.from_numpy(
         tf_weights['deepchem/models/tests/assets/TF_bias_T.npy'])   
         
    output_tensor = torch.from_numpy(
         np.load('deepchem/models/tests/assets/highway_output.npy').astype(
             np.float32))
  
    assert torch.allclose(output, output_tensor, atol=1e-04)



