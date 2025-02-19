import pytest
import numpy as np
try:
    import torch
    import deepchem.models.torch_models.chemnet_layers as layers
    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_Stem():
    """Test invoking stem layer."""

    input_np = np.array([[[[1.0, 0.5, 0.2, 0.8], [0.3, 0.4, 0.7, 0.2],
                           [0.6, 0.2, 0.9, 0.5], [0.4, 0.1, 0.5, 0.7]],
                          [[0.8, 0.3, 0.6, 0.9], [0.2, 0.9, 0.6, 0.5],
                           [0.7, 0.3, 0.8, 0.4], [0.8, 0.7, 0.3, 0.6]],
                          [[0.4, 0.7, 0.1, 0.2], [0.5, 0.1, 0.8, 0.8],
                           [0.8, 0.4, 0.1, 0.1], [0.6, 0.9, 0.2, 0.3]]]],
                        dtype=np.float32)

    input_torch = torch.tensor(input_np).permute(
        0, 1, 2, 3)  # Convert to (Batch, Channels, Height, Width)

    stem_torch = layers.Stem(in_channels=3, out_channels=4)

    fixed_kernel = np.full((4, 3, 4, 4), 0.05,
                           dtype=np.float32)  # Shape (out, in, h, w)
    fixed_bias = np.zeros((4,), dtype=np.float32)

    with torch.no_grad():
        stem_torch.conv_layer.weight = torch.nn.Parameter(
            torch.tensor(fixed_kernel))
        stem_torch.conv_layer.bias = torch.nn.Parameter(
            torch.tensor(fixed_bias))

    output_torch = stem_torch(input_torch).detach().numpy()

    output_tf = np.array([[[[1.2249998, 1.2249998, 1.2249998, 1.2249998]]]],
                         dtype=np.float32)
    output_tf = output_tf.transpose(
        0, 3, 1, 2)  # Convert to (Batch, Channels, Height, Width)

    assert output_torch.shape == output_tf.shape
    assert np.allclose(output_torch, output_tf, atol=1e-2)
