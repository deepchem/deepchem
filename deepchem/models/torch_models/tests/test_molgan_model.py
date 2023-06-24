import os
import unittest
import pytest
# import pandas as pd
# from deepchem.data import NumpyDataset
# from deepchem.feat.molecule_featurizers import MolGanFeaturizer
# from deepchem.models.optimizers import ExponentialDecay
try:
    # import torch
    from deepchem.models import BasicMolGANModel as MolGAN
    has_torch = True
except:
    has_torch = False


class test_molgan_model(unittest.TestCase):
    """
  Unit testing for MolGAN basic layers
  """

    @pytest.mark.torch
    def setUp(self):
        self.training_attempts = 6
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.vertices = 9
        self.nodes = 5
        self.edges = 5
        self.embedding_dim = 10
        self.dropout_rate = 0.0
        self.batch_size = 100
        self.first_convolution_unit = 128
        self.second_convolution_unit = 64
        self.aggregation_unit = 128
        self.model = MolGAN(edges=self.edges,
                            vertices=self.vertices,
                            nodes=self.nodes,
                            embedding_dim=self.embedding_dim,
                            dropout_rate=self.dropout_rate)

    @pytest.mark.torch
    def test_build(self):
        """
    Test if initialization data is set-up correctly
    """
        model = self.model
        assert model.batch_size == self.batch_size
        assert model.edges == self.edges
        assert model.nodes == self.nodes
        assert model.vertices == self.vertices
        assert model.dropout_rate == self.dropout_rate

    @pytest.mark.torch
    def test_helper_functions(self):
        """
    Check if helper functions are working correctly
    """
        model = self.model
        # test get_noise_input_shape
        assert model.get_noise_input_shape() == (self.embedding_dim,)
        # test get_data_input_shapes
        assert model.get_data_input_shapes() == [(self.vertices, self.vertices,
                                                  self.edges),
                                                 (self.vertices, self.nodes)]


if __name__ == '__main__':
    unittest.main()
