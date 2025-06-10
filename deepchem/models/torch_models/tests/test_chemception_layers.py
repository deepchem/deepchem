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
    """
    Test Stem layer against manually verified TensorFlow output
    using fixed weights and bias.
    """
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


@pytest.mark.torch
def test_InceptionResnetA():
    """
    Test InceptionResnetA layer against manually verified TensorFlow output
    using fixed weights and bias.
    """
    input_np = np.array(
        [[[[0.2712, 0.5123, 0.6565, 0.1889], [0.6174, 0.0609, 0.4317, 0.0775],
           [0.6640, 0.1586, 0.4666, 0.4570], [0.5987, 0.7010, 0.8050, 0.7374]],
          [[0.1532, 0.5063, 0.9928, 0.1148], [0.7192, 0.8554, 0.4936, 0.5310],
           [0.8861, 0.4392, 0.8616, 0.7651], [0.2985, 0.0640, 0.7019, 0.8535]],
          [[0.7230, 0.9014, 0.4116, 0.8261], [0.4230, 0.5786, 0.8521, 0.7080],
           [0.1534, 0.5855, 0.2708, 0.3537], [0.4066, 0.1785, 0.3009, 0.0765]]]
        ],
        dtype=np.float32)

    tf_outptut = np.array(
        [[[[25.955153, 25.837152, 26.406952], [41.345757, 41.33976, 41.73486],
           [41.573402, 41.909702, 41.3285], [25.353184, 25.279083, 25.990383]],
          [[40.21292, 40.31472, 40.01852], [63.13939, 63.933887, 63.65709],
           [64.421364, 64.48326, 64.84176], [39.861595, 40.315094, 40.492096]],
          [[38.42537, 38.64747, 37.91477], [60.783154, 61.063755, 61.210052],
           [63.015915, 63.410915, 62.820118], [39.95585, 40.263947, 39.852547]],
          [[23.25519, 22.95499, 23.063091], [37.369038, 36.732037, 36.84654],
           [39.336445, 39.233345, 38.832344], [25.398687, 25.514788, 24.737787]]
         ]],
        dtype=np.float32)

    tf_outptut = tf_outptut.transpose(
        0, 3, 1, 2)  # Convert to (Batch, Channels, Height, Width)

    input_torch = torch.tensor(input_np)

    InceptionResnetA_torch = layers.InceptionResnetAInceptionResnetA(
        in_channels=3, out_channels=32)

    for m in InceptionResnetA_torch.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.constant_(m.weight, 0.05)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    output_torch = InceptionResnetA_torch(input_torch).detach().numpy()

    assert output_torch.shape == tf_outptut.shape
    assert np.allclose(output_torch, tf_outptut, atol=1e-4)


@pytest.mark.torch
def test_InceptionResnetB():
    """
    Test InceptionResnetB layer against manually verified TensorFlow output
    using fixed weights and bias.
    """
    input_np = np.array([[[[0.983, 0.457, 0.008], [0.662, 0.623, 0.405],
                           [0.862, 0.3, 0.211], [0.169, 0.461, 0.367]],
                          [[0.401, 0.457, 0.898], [0.962, 0.556, 0.989],
                           [0.466, 0.403, 0.097], [0.832, 0.376, 0.483]],
                          [[0.425, 0.28, 0.33], [0.446, 0.357, 0.709],
                           [0.343, 0.876, 0.899], [0.624, 0.071, 0.498]],
                          [[0.901, 0.504, 0.256], [0.422, 0.551, 0.476],
                           [0.874, 0.634, 0.305], [0.007, 0.063, 0.393]]]],
                        dtype=np.float32)
    tf_outptut = np.array(
        [[[[10.1888895, 9.6628895, 9.21389], [9.887249, 9.848249, 9.630249],
           [10.06189, 9.499889, 9.41089], [9.338809, 9.63081, 9.536809]],
          [[9.63153, 9.68753, 10.12853], [10.252609, 9.846609, 10.27961],
           [9.633329, 9.57033, 9.26433], [10.057329, 9.60133, 9.708329]],
          [[9.59785, 9.452849, 9.50285], [9.65701, 9.56801, 9.92001],
           [9.60249, 10.13549, 10.15849], [9.809489, 9.25649, 9.68349]],
          [[10.123931, 9.726931, 9.47893], [9.627972, 9.756971, 9.681972],
           [10.109091, 9.869091, 9.5400915], [9.134091, 9.190091, 9.520091]]]],
        dtype=np.float32)

    tf_outptut = tf_outptut.transpose(
        0, 3, 1, 2)  # Convert to (Batch, Channels, Height, Width)

    input_torch = torch.tensor(input_np)

    InceptionResnetB_torch = layers.InceptionResnetB(in_channels=3,
                                                     out_channels=32)

    for m in InceptionResnetB_torch.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.constant_(m.weight, 0.05)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    output_torch = InceptionResnetB_torch(input_torch).detach().numpy()

    assert output_torch.shape == tf_outptut.shape
    assert np.allclose(output_torch, tf_outptut, atol=1e-4)
