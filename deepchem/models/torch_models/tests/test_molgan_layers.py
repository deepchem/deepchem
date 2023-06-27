import pytest

try:
    import torch
    # import torch.nn as nn
    import torch.nn.functional as F
    # , MolGANMultiConvolutionLayer, MolGANAggregationLayer, MolGANEncoderLayer
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

    assert output[0].shape == output_tf[0]  # adjacency_tensor
    assert output[1].shape == output_tf[1]  # node_tensor
    assert output[2].shape == output_tf[2]  # output of the layer

    assert output[0].shape == torch.Size([1, vertices, vertices,
                                          edges])  # adjacency_tensor
    assert output[1].shape == torch.Size([1, vertices, nodes])  # node_tensor
    assert output[2].shape == torch.Size([1, vertices,
                                          units])  # output of the layer
    assert layer.units == units
    assert layer.activation == F.tanh
    assert layer.edges == 5
    assert layer.dropout_rate == 0.0
