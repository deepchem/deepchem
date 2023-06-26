import unittest
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


class test_molgan_layers(unittest.TestCase):
    """
  Unit testing for MolGAN basic layers
  """

    @pytest.mark.torch
    def test_graph_convolution_layer(self):
        from deepchem.models.torch_models.layers import MolGANConvolutionLayer
        vertices = 9
        nodes = 5
        edges = 5
        units = 128

        layer = MolGANConvolutionLayer(units=units, edges=edges, nodes=nodes)
        adjacency_tensor = torch.randn((1, vertices, vertices, edges))
        node_tensor = torch.randn((1, vertices, nodes))
        output = layer([adjacency_tensor, node_tensor])

        assert output[0].shape == torch.Size([1, vertices, vertices,
                                              edges])  # adjacency_tensor
        assert output[1].shape == torch.Size([1, vertices,
                                              nodes])  # node_tensor
        assert output[2].shape == torch.Size([1, vertices,
                                              units])  # output of the layer
        assert layer.units == units
        assert layer.activation == F.tanh
        assert layer.edges == 5
        assert layer.dropout_rate == 0.0


if __name__ == '__main__':
    unittest.main()
