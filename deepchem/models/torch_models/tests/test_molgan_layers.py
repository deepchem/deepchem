import pytest

try:
    import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    import numpy as np
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_graph_convolution_layer():
    from deepchem.models.torch_models.layers import MolGANConvolutionLayer
    vertices = 9
    nodes = 5
    edges = 5
    units = 128

    layer = MolGANConvolutionLayer(units=units, edges=edges, nodes=nodes)
    adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    node_tensor = torch.randn((1, vertices, nodes))
    output = layer([adjacency_tensor, node_tensor])
    output_tf = [
        (1, 9, 9, 5), (1, 9, 5), (1, 9, 128)
    ]  # None has been converted to 1 as batch size is taken as 1 in torch

    # Testing Shapes
    assert output[0].shape == output_tf[0]  # adjacency_tensor
    assert output[1].shape == output_tf[1]  # node_tensor
    assert output[2].shape == output_tf[2]  # output of the layer

    assert output[0].shape == torch.Size([1, vertices, vertices,
                                          edges])  # adjacency_tensor
    assert output[1].shape == torch.Size([1, vertices, nodes])  # node_tensor
    assert output[2].shape == torch.Size([1, vertices,
                                          units])  # output of the layer

    # Testing values
    assert layer.units == units
    assert layer.activation == torch.tanh
    assert layer.edges == 5
    assert layer.dropout_rate == 0.0


@pytest.mark.torch
def test_graph_convolution_layer_values():
    from deepchem.models.torch_models.layers import MolGANConvolutionLayer
    vertices = 9
    nodes = 5
    edges = 5
    units = 128

    torch.manual_seed(21)  # Setting seed for reproducibility
    layer = MolGANConvolutionLayer(units=units, edges=edges, nodes=nodes)
    tf_weights = np.load(
        'deepchem/models/tests/assets/molgan_conv_layer_weights.npy',
        allow_pickle=True).item()
    with torch.no_grad():
        for idx, dense in enumerate(layer.dense1):
            # Dense1 is a list of dense layers
            weight_name = f'layer1/dense_{idx+4}/kernel:0'
            bias_name = f'layer1/dense_{idx+4}/bias:0'
            dense.weight.data = torch.from_numpy(
                np.transpose(tf_weights[weight_name]))
            dense.bias.data = torch.from_numpy(tf_weights[bias_name])

    layer.dense2.weight.data = torch.from_numpy(
        np.transpose(tf_weights['layer1/dense_8/kernel:0']))
    layer.dense2.bias.data = torch.from_numpy(
        tf_weights['layer1/dense_8/bias:0'])
    adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    node_tensor = torch.randn((1, vertices, nodes))
    output = layer([adjacency_tensor, node_tensor])

    adjacency_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_adj_tensor.npy').astype(
            np.float32))
    node_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_nod_tensor.npy').astype(
            np.float32))
    output = layer([adjacency_tensor, node_tensor])
    output_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_conv_layer_op.npy').astype(
            np.float32))

    # Testing Values
    assert torch.allclose(output[0], adjacency_tensor, atol=1e-06)
    assert torch.allclose(output[1], node_tensor, atol=1e-06)
    assert torch.allclose(output[2], output_tensor, atol=1e-04)


@pytest.mark.torch
def test_aggregation_layer_shape():
    from deepchem.models.torch_models.layers import MolGANAggregationLayer
    vertices = 9
    units = 128

    layer = MolGANAggregationLayer(units=units)
    hidden_tensor = torch.randn((1, vertices, units))
    output = layer(hidden_tensor)
    output_tf = (
        1, 128
    )  # None has been converted to 1 as batch size is taken as 1 in torch

    # Testing Shapes with TF Model Output
    assert output.shape == output_tf

    # Testing Shapes
    assert output.shape == (1, units)
    assert layer.units == units
    assert layer.activation == torch.tanh
    assert layer.dropout_rate == 0.0


@pytest.mark.torch
def test_aggregation_layer_values():
    from deepchem.models.torch_models.layers import MolGANAggregationLayer
    units = 128

    torch.manual_seed(21)  # Setting seed for reproducibility
    layer = MolGANAggregationLayer(units=units, name='layer1')
    tf_weights = np.load(
        'deepchem/models/tests/assets/molgan_agg_layer_weights.npy',
        allow_pickle=True).item()
    with torch.no_grad():
        layer.d1.weight.data = torch.from_numpy(
            np.transpose(tf_weights['layer1/dense_27/kernel:0']))
        layer.d1.bias.data = torch.from_numpy(
            tf_weights['layer1/dense_27/bias:0'])
        layer.d2.weight.data = torch.from_numpy(
            np.transpose(tf_weights['layer1/dense_28/kernel:0']))
        layer.d2.bias.data = torch.from_numpy(
            tf_weights['layer1/dense_28/bias:0'])

    hidden_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_agg_tensor.npy').astype(
            np.float32))
    output = layer(hidden_tensor)
    output_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_agg_layer_op.npy').astype(
            np.float32))

    # Testing Values
    assert torch.allclose(output, output_tensor, atol=1e-04)


@pytest.mark.torch
def test_multigraph_convolution_layer_shape():
    from deepchem.models.torch_models.layers import MolGANMultiConvolutionLayer
    vertices = 9
    nodes = 5
    edges = 5
    first_convolution_unit = 128
    second_convolution_unit = 64
    units = [first_convolution_unit, second_convolution_unit]

    layer = MolGANMultiConvolutionLayer(units=units, edges=edges)
    adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    node_tensor = torch.randn((1, vertices, nodes))
    model = layer([adjacency_tensor, node_tensor])

    assert model.shape == (1, vertices, second_convolution_unit)
    assert layer.units == units
    assert layer.activation == torch.tanh
    assert layer.edges == 5
    assert layer.dropout_rate == 0.0


@pytest.mark.torch
def test_multigraph_convolution_layer_values():
    from deepchem.models.torch_models.layers import MolGANMultiConvolutionLayer
    nodes = 5
    edges = 5
    first_convolution_unit = 128
    second_convolution_unit = 64
    units = [first_convolution_unit, second_convolution_unit]

    torch.manual_seed(21)  # Setting seed for reproducibility
    layer_multi_conv = MolGANMultiConvolutionLayer(units=units,
                                                   nodes=nodes,
                                                   edges=edges,
                                                   name='layer1')
    tf_weights = np.load(
        'deepchem/models/tests/assets/molgan_multi_conv_layer_weights.npy',
        allow_pickle=True).item()
    with torch.no_grad():
        x = 10
        # testing first convolution layer

        # dense1 layer - list of dense layers
        for idx, dense in enumerate(layer_multi_conv.first_convolution.dense1):
            weight_name = f'layer1//dense_{idx+x}/kernel:0'
            bias_name = f'layer1//dense_{idx+x}/bias:0'
            dense.weight.data = torch.from_numpy(
                np.transpose(tf_weights[weight_name]))
            dense.bias.data = torch.from_numpy(tf_weights[bias_name])
        idx += 1
        # dense2 layer - single dense layer
        layer_multi_conv.first_convolution.dense2.weight.data = torch.from_numpy(
            np.transpose(tf_weights[f'layer1//dense_{idx+x}/kernel:0']))
        layer_multi_conv.first_convolution.dense2.bias.data = torch.from_numpy(
            tf_weights[f'layer1//dense_{idx+x}/bias:0'])
        x += 5

        # testing rest of the convolution layer
        for idx_, layer in enumerate(layer_multi_conv.gcl):
            # dense1 layer - list of dense layers
            for idx, dense in enumerate(layer.dense1):
                weight_name = f'layer1//dense_{idx+x}/kernel:0'
                bias_name = f'layer1//dense_{idx+x}/bias:0'
                dense.weight.data = torch.from_numpy(
                    np.transpose(tf_weights[weight_name]))
                dense.bias.data = torch.from_numpy(tf_weights[bias_name])
            x += 1
            # dense2 layer - single dense layer
            layer.dense2.weight.data = torch.from_numpy(
                np.transpose(tf_weights[f'layer1//dense_{idx+x}/kernel:0']))
            layer.dense2.bias.data = torch.from_numpy(
                tf_weights[f'layer1//dense_{idx+x}/bias:0'])

    # Loading input tensors
    adjacency_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_adj_tensor.npy').astype(
            np.float32))
    node_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_nod_tensor.npy').astype(
            np.float32))

    # Testing output
    output = layer_multi_conv([adjacency_tensor, node_tensor])
    output_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_multi_conv_layer_op.npy').
        astype(np.float32))
    assert torch.allclose(output, output_tensor, atol=1e-04)


@pytest.mark.torch
def test_graph_encoder_layer_shape():
    from deepchem.models.torch_models.layers import MolGANEncoderLayer
    vertices = 9
    nodes = 5
    edges = 5
    first_convolution_unit = 128
    second_convolution_unit = 64
    aggregation_unit = 128
    units = [(first_convolution_unit, second_convolution_unit),
             aggregation_unit]

    layer = MolGANEncoderLayer(units=units, edges=edges)
    adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    node_tensor = torch.randn((1, vertices, nodes))
    model = layer([adjacency_tensor, node_tensor])

    assert model.shape == (1, aggregation_unit)
    assert layer.graph_convolution_units == (first_convolution_unit,
                                             second_convolution_unit)
    assert layer.auxiliary_units == aggregation_unit
    assert layer.activation == torch.tanh
    assert layer.edges == 5
    assert layer.dropout_rate == 0.0


@pytest.mark.torch
def test_graph_encoder_layer_values():
    """
    Test to check the Values of the Graph Encoder Layer
    It first loads the weights of the TF model
    Then it starts transfering the weights to the torch model
        1. MultiConvolution Layer
            1.1 First Convolution Layer
            1.2 Rest of the Convolution Layers
        2. Aggregation Layer
    Then it loads the input tensors and checks the output
    """
    from deepchem.models.torch_models.layers import MolGANEncoderLayer
    nodes = 5
    edges = 5
    first_convolution_unit = 128
    second_convolution_unit = 64
    aggregation_unit = 128
    units = [(first_convolution_unit, second_convolution_unit),
             aggregation_unit]

    torch.manual_seed(21)
    tf_weights = np.load(
        'deepchem/models/tests/assets/molgan_encoder_layer_weights.npy',
        allow_pickle=True).item()
    torch_model_encoder = MolGANEncoderLayer(units=units,
                                             nodes=nodes,
                                             edges=edges,
                                             name='layer1')

    x = 12  # the starting number for the dense layers in the tf model weights
    with torch.no_grad():
        # Testing MultiConvolution Layer

        # Testing First Convolution Layer
        # dense1 layer - list of dense layers
        for idx, dense in enumerate(
                torch_model_encoder.multi_graph_convolution_layer.
                first_convolution.dense1):
            weight_name = f'layer1///dense_{idx+x}/kernel:0'
            bias_name = f'layer1///dense_{idx+x}/bias:0'

            dense.weight.data = torch.from_numpy(
                np.transpose(tf_weights[weight_name]))
            dense.bias.data = torch.from_numpy(tf_weights[bias_name])
        idx += 1

        # dense2 layer - single dense layer
        torch_model_encoder.multi_graph_convolution_layer.first_convolution.dense2.weight.data = torch.from_numpy(
            np.transpose(tf_weights[f'layer1///dense_{idx+x}/kernel:0']))
        torch_model_encoder.multi_graph_convolution_layer.first_convolution.dense2.bias.data = torch.from_numpy(
            tf_weights[f'layer1///dense_{idx+x}/bias:0'])
        x += 5

        # Testing rest of the Multi convolution layer
        for idx_, layer in enumerate(
                torch_model_encoder.multi_graph_convolution_layer.gcl):
            # dense1 layer - list of dense layers
            for idx, dense in enumerate(layer.dense1):
                weight_name = f'layer1///dense_{idx+x}/kernel:0'
                bias_name = f'layer1///dense_{idx+x}/bias:0'
                dense.weight.data = torch.from_numpy(
                    np.transpose(tf_weights[weight_name]))
                dense.bias.data = torch.from_numpy(tf_weights[bias_name])
            x += 1

            # dense2 layer - single dense layer
            layer.dense2.weight.data = torch.from_numpy(
                np.transpose(tf_weights[f'layer1///dense_{idx+x}/kernel:0']))
            layer.dense2.bias.data = torch.from_numpy(
                tf_weights[f'layer1///dense_{idx+x}/bias:0'])

        # Testing Aggregation Layer
        torch_model_encoder.graph_aggregation_layer.d1.weight.data = torch.from_numpy(
            np.transpose(tf_weights['layer1//dense_22/kernel:0']))
        torch_model_encoder.graph_aggregation_layer.d1.bias.data = torch.from_numpy(
            tf_weights['layer1//dense_22/bias:0'])
        torch_model_encoder.graph_aggregation_layer.d2.weight.data = torch.from_numpy(
            np.transpose(tf_weights['layer1//dense_23/kernel:0']))
        torch_model_encoder.graph_aggregation_layer.d2.bias.data = torch.from_numpy(
            tf_weights['layer1//dense_23/bias:0'])

    # Loading input tensors
    adjacency_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_adj_tensor.npy').astype(
            np.float32))
    node_tensor = torch.from_numpy(
        np.load('deepchem/models/tests/assets/molgan_nod_tensor.npy').astype(
            np.float32))

    # Testing output
    output = torch_model_encoder([adjacency_tensor, node_tensor])
    output_tensor = torch.from_numpy(
        np.load(
            'deepchem/models/tests/assets/molgan_encoder_layer_op.npy').astype(
                np.float32))
    assert torch.allclose(output, output_tensor, atol=1e-04)
