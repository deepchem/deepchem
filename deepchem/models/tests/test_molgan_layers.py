import unittest
import pytest

try:
    from tensorflow import keras
    from tensorflow.keras.layers import Input
    from tensorflow.keras import activations
    from deepchem.models.layers import MolGANConvolutionLayer, MolGANMultiConvolutionLayer, MolGANAggregationLayer, MolGANEncoderLayer
    has_tensorflow = True
except:
    has_tensorflow = False


class test_molgan_layers(unittest.TestCase):
    """
  Unit testing for MolGAN basic layers
  """

    @pytest.mark.tensorflow
    def test_graph_convolution_layer(self):
        vertices = 9
        nodes = 5
        edges = 5
        units = 128

        layer = MolGANConvolutionLayer(units=units, edges=edges)
        adjacency_tensor = Input(shape=(vertices, vertices, edges))
        node_tensor = Input(shape=(vertices, nodes))
        output = layer([adjacency_tensor, node_tensor])
        model = keras.Model(inputs=[adjacency_tensor, node_tensor],
                            outputs=[output])

        assert model.output_shape == [((None, vertices, vertices, edges),
                                       (None, vertices, nodes), (None, vertices,
                                                                 units))]
        assert layer.units == units
        assert layer.activation == activations.tanh
        assert layer.edges == 5
        assert layer.dropout_rate == 0.0

    @pytest.mark.tensorflow
    def test_aggregation_layer(self):
        vertices = 9
        units = 128

        layer = MolGANAggregationLayer(units=units)
        hidden_tensor = Input(shape=(vertices, units))
        output = layer(hidden_tensor)
        model = keras.Model(inputs=[hidden_tensor], outputs=[output])

        assert model.output_shape == (None, units)
        assert layer.units == units
        assert layer.activation == activations.tanh
        assert layer.dropout_rate == 0.0

    @pytest.mark.tensorflow
    def test_multigraph_convolution_layer(self):
        vertices = 9
        nodes = 5
        edges = 5
        first_convolution_unit = 128
        second_convolution_unit = 64
        units = [first_convolution_unit, second_convolution_unit]

        layer = MolGANMultiConvolutionLayer(units=units, edges=edges)
        adjacency_tensor = Input(shape=(vertices, vertices, edges))
        node_tensor = Input(shape=(vertices, nodes))
        hidden_tensor = layer([adjacency_tensor, node_tensor])
        model = keras.Model(inputs=[adjacency_tensor, node_tensor],
                            outputs=[hidden_tensor])

        assert model.output_shape == (None, vertices, second_convolution_unit)
        assert layer.units == units
        assert layer.activation == activations.tanh
        assert layer.edges == 5
        assert layer.dropout_rate == 0.0

    @pytest.mark.tensorflow
    def test_graph_encoder_layer(self):
        vertices = 9
        nodes = 5
        edges = 5
        first_convolution_unit = 128
        second_convolution_unit = 64
        aggregation_unit = 128
        units = [(first_convolution_unit, second_convolution_unit),
                 aggregation_unit]

        layer = MolGANEncoderLayer(units=units, edges=edges)
        adjacency_tensor = Input(shape=(vertices, vertices, edges))
        node_tensor = Input(shape=(vertices, nodes))
        output = layer([adjacency_tensor, node_tensor])
        model = keras.Model(inputs=[adjacency_tensor, node_tensor],
                            outputs=[output])

        assert model.output_shape == (None, aggregation_unit)
        assert layer.graph_convolution_units == (first_convolution_unit,
                                                 second_convolution_unit)
        assert layer.auxiliary_units == aggregation_unit
        assert layer.activation == activations.tanh
        assert layer.edges == 5
        assert layer.dropout_rate == 0.0


if __name__ == '__main__':
    unittest.main()
