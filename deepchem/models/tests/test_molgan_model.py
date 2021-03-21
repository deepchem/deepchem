import os
import unittest

import pandas as pd
from deepchem.data import NumpyDataset
from deepchem.feat.molecule_featurizers import MolGanFeaturizer
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
from tensorflow import one_hot
from tensorflow.errors import InternalError
from tensorflow.keras.backend import clear_session as keras_clear_session


class test_molgan_model(unittest.TestCase):
  """
  Unit testing for MolGAN basic layers
  """

  def setUp(self):
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

  def test_training(self):
    """
    Check training of the basicMolGANmodel on small number of compounds.
    Due to training instability try a few times and see if it worked at least once.
    Typically it fails between 1-3 times of 10.
    This is something that needs to be addressed in future releases.
    """

    input_file = os.path.join(self.current_dir, "molgan_example.csv")
    data = pd.read_csv(input_file)
    molecules = list(data['Molecule'])
    feat = MolGanFeaturizer()
    featurized = feat.featurize(molecules)
    dataset = NumpyDataset([x.adjacency_matrix for x in featurized],
                           [x.node_features for x in featurized])
    valid_attempts = 0
    for _ in range(10):
      # try to catch tensorflow internal errors
      try:
        # force clear tensor flow backend
        keras_clear_session()
        # create new model
        gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000))

        # generate input
        def iterbatches(epochs):
          for __ in range(epochs):
            for batch in dataset.iterbatches(
                    batch_size=gan.batch_size, pad_batches=True):
              adjacency_tensor = one_hot(batch[0], gan.edges)
              node_tesor = one_hot(batch[1], gan.nodes)

              yield {
                  gan.data_inputs[0]: adjacency_tensor,
                  gan.data_inputs[1]: node_tesor
              }

        # train model
        gan.fit_gan(
            iterbatches(1000), generator_steps=0.2, checkpoint_interval=0)

        # generate sample
        g = gan.predict_gan_generator(1000)
        # check how many valid molecules were created and add to list
        generated_molecules = feat.defeaturize(g)
        valid_molecules_count = len(
            list(filter(lambda x: x is not None, generated_molecules)))
        if valid_molecules_count:
          valid_attempts = valid_attempts + 1
      except InternalError:
        print(
            'Tensor flow internal error raised. Make sure no other instance of tensorflow e.g. jupyter notebook, is running.'
        )
        break

    assert valid_attempts > 0


if __name__ == '__main__':
  unittest.main()
