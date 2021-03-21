import unittest

from deepchem.models import BasicMolGANModel as MolGAN


class test_molgan_model(unittest.TestCase):
  """
  Unit testing for MolGAN basic layers
  """

  def setUp(self):
    self.vertices = 9
    self.nodes = 5
    self.edges = 5
    self.embedding_dim = 10
    self.dropout_rate = 0.0
    self.batch_size = 100
    self.first_convolution_unit = 128
    self.second_convolution_unit = 64
    self.aggregation_unit = 128
    self.model = MolGAN(
        edges=self.edges,
        vertices=self.vertices,
        nodes=self.nodes,
        embedding_dim=self.embedding_dim,
        dropout_rate=self.dropout_rate)

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
    assert len(model.generators) == 1
    assert len(model.discriminators) == 1

  def test_shapes(self):
    """
    Check if input and output shapes are correct
    """
    model = self.model

    # test if adjacency matrix input is correctly set
    assert model.discriminators[0].input_shape[0] == (None, self.vertices,
                                                      self.vertices, self.edges)
    # test if nodes features matrix input is correctly set
    assert model.discriminators[0].input_shape[1] == (None, self.vertices,
                                                      self.edges)
    # check discriminator shape
    assert model.discriminators[0].output_shape == (None, 1)
    # check training edges logits shape
    assert model.generators[0].output_shape[0] == (None, self.vertices,
                                                   self.vertices, self.edges)
    # check training nodes logits shapes
    assert model.generators[0].output_shape[1] == (None, self.vertices,
                                                   self.nodes)
    # check molecule generation edges logits shapes
    assert model.generators[0].output_shape[2] == (None, self.vertices,
                                                   self.vertices)
    # check molecule generation nodes logits shapes
    assert model.generators[0].output_shape[3] == (None, self.vertices)


if __name__ == '__main__':
  unittest.main()
