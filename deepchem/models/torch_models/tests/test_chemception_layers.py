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

    InceptionResnetA_torch = layers.InceptionResnetA(in_channels=3,
                                                     out_channels=32)

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
    input_np = np.array(
        [[[[0.9830, 0.6620, 0.8620, 0.1690], [0.4010, 0.9620, 0.4660, 0.8320],
           [0.4250, 0.4460, 0.3430, 0.6240], [0.9010, 0.4220, 0.8740, 0.0070]],
          [[0.4570, 0.6230, 0.3000, 0.4610], [0.4570, 0.5560, 0.4030, 0.3760],
           [0.2800, 0.3570, 0.8760, 0.0710], [0.5040, 0.5510, 0.6340, 0.0630]],
          [[0.0080, 0.4050, 0.2110, 0.3670], [0.8980, 0.9890, 0.0970, 0.4830],
           [0.3300, 0.7090, 0.8990, 0.4980], [0.2560, 0.4760, 0.3050, 0.3930]]]
        ],
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


@pytest.mark.torch
def test_InceptionResnetC():
    """
    Test InceptionResnetC layer against manually verified TensorFlow output
    using fixed weights and bias.
    """
    input_np = np.array(
        [[[[0.3210, 0.6150, 0.2950, 0.6300], [0.4900, 0.5370, 0.8750, 0.0190],
           [0.6870, 0.8710, 0.6930, 0.8690], [0.3030, 0.7260, 0.9150, 0.8570]],
          [[0.5170, 0.6210, 0.0540, 0.5840], [0.8630, 0.2740, 0.9280, 0.0430],
           [0.2650, 0.3280, 0.2720, 0.2950], [0.9850, 0.4980, 0.2200, 0.6490]],
          [[0.5950, 0.2920, 0.0940, 0.5650], [0.4570, 0.7400, 0.2830, 0.8070],
           [0.8800, 0.1880, 0.4780, 0.7540], [0.7420, 0.6380, 0.4570, 0.6810]]]
        ],
        dtype=np.float32)

    tf_output = np.array(
        [[[[2.4005194, 2.5965192, 2.6745193], [3.488132, 3.494132, 3.165132],
           [2.896405, 2.6554048, 2.695405], [2.3813322, 2.3353322, 2.316332]],
          [[3.6001463, 3.9731462, 3.5671463], [4.8609247, 4.5979247, 5.063925],
           [5.083527, 5.136527, 4.491527], [2.742129, 2.766129, 3.530129]],
          [[4.088258, 3.6662579, 4.281258], [5.828266, 5.2852664, 5.1452665],
           [5.437809, 5.016809, 5.2228093], [4.159965, 3.5859652, 4.0449653]],
          [[2.6755004, 3.3575006, 3.1145005], [4.0283413, 3.8003411, 3.9403412],
           [4.271264, 3.5762641, 3.8132641], [3.2510748, 3.0430746, 3.0750747]]]
        ],
        dtype=np.float32)

    tf_output = tf_output.transpose(
        0, 3, 1, 2)  # Convert to (Batch, Channels, Height, Width)

    input_torch = torch.tensor(input_np)

    InceptionResnetC_torch = layers.InceptionResnetC(in_channels=3,
                                                     out_channels=32)

    for m in InceptionResnetC_torch.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.constant_(m.weight, 0.05)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    output_torch = InceptionResnetC_torch(input_torch).detach().numpy()

    assert output_torch.shape == tf_output.shape
    assert np.allclose(output_torch, tf_output, atol=1e-4)


@pytest.mark.torch
def test_ReductionA():
    """
    Test ReductionA layer against manually verified TensorFlow output
    using fixed weights and bias.
    """
    input_np = np.array(
        [[[[0.3210, 0.6150, 0.2950, 0.6300], [0.4900, 0.5370, 0.8750, 0.0190],
           [0.6870, 0.8710, 0.6930, 0.8690], [0.3030, 0.7260, 0.9150, 0.8570]],
          [[0.5170, 0.6210, 0.0540, 0.5840], [0.8630, 0.2740, 0.9280, 0.0430],
           [0.2650, 0.3280, 0.2720, 0.2950], [0.9850, 0.4980, 0.2200, 0.6490]],
          [[0.5950, 0.2920, 0.0940, 0.5650], [0.4570, 0.7400, 0.2830, 0.8070],
           [0.8800, 0.1880, 0.4780, 0.7540], [0.7420, 0.6380, 0.4570, 0.6810]]]
        ],
        dtype=np.float32)

    tf_output = np.array([[[[
        0.875, 0.928, 0.88, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565,
        0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565,
        0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565,
        0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565,
        0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565,
        0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565, 0.67565,
        0.67565, 0.67565, 0.67565, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216, 12.857216, 12.857216, 12.857216, 12.857216,
        12.857216, 12.857216
    ]]]],
                         dtype=np.float32)

    tf_output = tf_output.transpose(
        0, 3, 1, 2)  # Convert to (Batch, Channels, Height, Width)

    input_torch = torch.from_numpy(input_np)

    model = layers.ReductionA(in_channels=3, out_channels=32)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.constant_(m.weight, 0.05)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    output_torch = model(input_torch).detach().numpy()

    assert output_torch.shape == tf_output.shape
    assert np.allclose(output_torch, tf_output, atol=1e-4)


@pytest.mark.torch
def test_ReductionB():
    """
    Test ReductionB layer against manually verified TensorFlow output
    using fixed weights and bias.
    """
    input_np = np.array(
        [[[[0.3210, 0.6150, 0.2950, 0.6300], [0.4900, 0.5370, 0.8750, 0.0190],
           [0.6870, 0.8710, 0.6930, 0.8690], [0.3030, 0.7260, 0.9150, 0.8570]],
          [[0.5170, 0.6210, 0.0540, 0.5840], [0.8630, 0.2740, 0.9280, 0.0430],
           [0.2650, 0.3280, 0.2720, 0.2950], [0.9850, 0.4980, 0.2200, 0.6490]],
          [[0.5950, 0.2920, 0.0940, 0.5650], [0.4570, 0.7400, 0.2830, 0.8070],
           [0.8800, 0.1880, 0.4780, 0.7540], [0.7420, 0.6380, 0.4570, 0.6810]]]
        ],
        dtype=np.float32)

    tf_output = np.array([[[[
        0.875, 0.928, 0.88, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398, 1.0810398,
        1.0810398, 1.0810398, 5.4658103, 5.4658103, 5.4658103, 5.4658103,
        5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103,
        5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103,
        5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103,
        5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103,
        5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103,
        5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103, 5.4658103
    ]]]],
                         dtype=np.float32)

    tf_output = tf_output.transpose(
        0, 3, 1, 2)  # Convert to (Batch, Channels, Height, Width)

    input_torch = torch.from_numpy(input_np)

    model = layers.ReductionB(in_channels=3, out_channels=32)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.constant_(m.weight, 0.05)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    output_torch = model(input_torch).detach().numpy()

    assert output_torch.shape == tf_output.shape
    assert np.allclose(output_torch, tf_output, atol=1e-4)
