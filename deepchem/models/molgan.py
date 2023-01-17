from typing import Optional, List, Tuple, Any

import tensorflow as tf
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
from deepchem.models import WGAN
from deepchem.models.layers import MolGANEncoderLayer
from tensorflow import keras
from tensorflow.keras import layers


class BasicMolGANModel(WGAN):
  """
  Model for de-novo generation of small molecules based on work of Nicola De Cao et al. [1]_.
  Utilizes WGAN infrastructure; uses adjacency matrix and node features as inputs.
  Inputs need to be one-hot representation.

  Examples
  --------
  >>>
  >> import deepchem as dc
  >> from deepchem.models import BasicMolGANModel as MolGAN
  >> from deepchem.models.optimizers import ExponentialDecay
  >> from tensorflow import one_hot
  >> smiles = ['CCC', 'C1=CC=CC=C1', 'CNC' ]
  >> # create featurizer
  >> feat = dc.feat.MolGanFeaturizer()
  >> # featurize molecules
  >> features = feat.featurize(smiles)
  >> # Remove empty objects
  >> features = list(filter(lambda x: x is not None, features))
  >> # create model
  >> gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000))
  >> dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features],[x.node_features for x in features])
  >> def iterbatches(epochs):
  >>     for i in range(epochs):
  >>         for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
  >>             adjacency_tensor = one_hot(batch[0], gan.edges)
  >>             node_tensor = one_hot(batch[1], gan.nodes)
  >>             yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tensor}
  >> gan.fit_gan(iterbatches(8), generator_steps=0.2, checkpoint_interval=5000)
  >> generated_data = gan.predict_gan_generator(1000)
  >> # convert graphs to RDKitmolecules
  >> nmols = feat.defeaturize(generated_data)
  >> print("{} molecules generated".format(len(nmols)))
  >> # remove invalid moles
  >> nmols = list(filter(lambda x: x is not None, nmols))
  >> # currently training is unstable so 0 is a common outcome
  >> print ("{} valid molecules".format(len(nmols)))

  References
  ----------
  .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model 
         for small molecular graphs", https://arxiv.org/abs/1805.11973
  """

  def __init__(self,
               edges: int = 5,
               vertices: int = 9,
               nodes: int = 5,
               embedding_dim: int = 10,
               dropout_rate: float = 0.0,
               **kwargs):
    """
    Initialize the model

    Parameters
    ----------
    edges: int, default 5
        Number of bond types includes BondType.Zero
    vertices: int, default 9
        Max number of atoms in adjacency and node features matrices
    nodes: int, default 5
        Number of atom types in node features matrix
    embedding_dim: int, default 10
        Size of noise input array
    dropout_rate: float, default = 0.
        Rate of dropout used across whole model
    name: str, default ''
        Name of the model
    """

    self.edges = edges
    self.vertices = vertices
    self.nodes = nodes
    self.embedding_dim = embedding_dim
    self.dropout_rate = dropout_rate

    super(BasicMolGANModel, self).__init__(**kwargs)

  def get_noise_input_shape(self) -> Tuple[int]:
    """
    Return shape of the noise input used in generator

    Returns
    -------
    Tuple
        Shape of the noise input
    """

    return (self.embedding_dim,)

  def get_data_input_shapes(self) -> List:
    """
    Return input shape of the discriminator

    Returns
    -------
    List
        List of shapes used as an input for distriminator.
    """
    return [
        (self.vertices, self.vertices, self.edges),
        (self.vertices, self.nodes),
    ]

  def create_generator(self) -> keras.Model:
    """
    Create generator model.
    Take noise data as an input and processes it through number of
    dense and dropout layers. Then data is converted into two forms
    one used for training and other for generation of compounds.
    The model has two outputs:
      1. edges
      2. nodes
    The format differs depending on intended use (training or sample generation).
    For sample generation use flag, sample_generation=True while calling generator
    i.e. gan.generators[0](noise_input, training=False, sample_generation=True).
    In case of training, not flag is necessary.
    """
    return BasicMolGANGenerator(
        vertices=self.vertices,
        edges=self.edges,
        nodes=self.nodes,
        dropout_rate=self.dropout_rate,
        embedding_dim=self.embedding_dim)

  def create_discriminator(self) -> keras.Model:
    """
    Create discriminator model based on MolGAN layers.
    Takes two inputs:
      1. adjacency tensor, containing bond information
      2. nodes tensor, containing atom information
    The input vectors need to be in one-hot encoding format.
    Use MolGAN featurizer for that purpose. It will be simplified
    in the future release.
    """
    adjacency_tensor = layers.Input(
        shape=(self.vertices, self.vertices, self.edges))
    node_tensor = layers.Input(shape=(self.vertices, self.nodes))

    graph = MolGANEncoderLayer(
        units=[(128, 64), 128],
        dropout_rate=self.dropout_rate,
        edges=self.edges)([adjacency_tensor, node_tensor])
    dense = layers.Dense(units=128, activation="tanh")(graph)
    dense = layers.Dropout(self.dropout_rate)(dense)
    dense = layers.Dense(units=64, activation="tanh")(dense)
    dense = layers.Dropout(self.dropout_rate)(dense)
    output = layers.Dense(units=1)(dense)

    return keras.Model(
        inputs=[
            adjacency_tensor,
            node_tensor,
        ], outputs=[output])

  def predict_gan_generator(self,
                            batch_size: int = 1,
                            noise_input: Optional[List] = None,
                            conditional_inputs: List = [],
                            generator_index: int = 0) -> List[GraphMatrix]:
    """
    Use the GAN to generate a batch of samples.

    Parameters
    ----------
    batch_size: int
      the number of samples to generate.  If either noise_input or
      conditional_inputs is specified, this argument is ignored since the batch
      size is then determined by the size of that argument.
    noise_input: array
      the value to use for the generator's noise input.  If None (the default),
      get_noise_batch() is called to generate a random input, so each call will
      produce a new set of samples.
    conditional_inputs: list of arrays
      NOT USED.
      the values to use for all conditional inputs.  This must be specified if
      the GAN has any conditional inputs.
    generator_index: int
      NOT USED.
      the index of the generator (between 0 and n_generators-1) to use for
      generating the samples.

    Returns
    -------
    List[GraphMatrix]
      Returns a list of GraphMatrix object that can be converted into
      RDKit molecules using MolGANFeaturizer defeaturize function.
    """

    if noise_input is not None:
      batch_size = len(noise_input)
    if noise_input is None:
      noise_input = self.get_noise_batch(batch_size)
    print(f"Generating {batch_size} samples")
    adjacency_matrix, nodes_features = self.generators[0](
        noise_input, training=False, sample_generation=True)
    graphs = [
        GraphMatrix(i, j)
        for i, j in zip(adjacency_matrix.numpy(), nodes_features.numpy())
    ]
    return graphs


class BasicMolGANGenerator(tf.keras.Model):
  """
  Generator class for BasicMolGAN model.
  Using subclassing rather than functional API due to requirement
  to swap between two outputs depending on situation.
  In order to get output that used for sample generation
  (conversion to rdkit molecules) pass sample_generation=True argument while
  calling the model i.e. adjacency_matrix, nodes_features = self.generators[0](
  noise_input, training=False, sample_generation=True)
  This is automatically done in predict_gan_generator().
  """

  def __init__(self,
               vertices: int = 9,
               edges: int = 5,
               nodes: int = 5,
               dropout_rate: float = 0.,
               embedding_dim: int = 10,
               name: str = "SimpleMolGANGenerator",
               **kwargs):
    """
    Initialize model.

    Parameters
    ----------
    vertices : int, optional
        number of max atoms dataset molecules (incl. empty atom), by default 9
    edges : int, optional
        number of bond types in molecules, by default 5
    nodes : int, optional
        number of atom types in molecules, by default 5
    dropout_rate : float, optional
        rate of dropout, by default 0.
    embedding_dim : int, optional
        noise input dimensions, by default 10
    name : str, optional
        name of the model, by default "SimpleMolGANGenerator"
    """
    super(BasicMolGANGenerator, self).__init__(name=name, **kwargs)
    self.vertices = vertices
    self.edges = edges
    self.nodes = nodes
    self.dropout_rate = dropout_rate
    self.embedding_dim = embedding_dim

    self.dense1 = layers.Dense(
        128, activation="tanh", input_shape=(self.embedding_dim,))
    self.dropout1 = layers.Dropout(self.dropout_rate)
    self.dense2 = layers.Dense(256, activation="tanh")
    self.dropout2 = layers.Dropout(self.dropout_rate)
    self.dense3 = layers.Dense(512, activation="tanh")
    self.dropout3 = layers.Dropout(self.dropout_rate)

    # edges logits used during training
    self.edges_dense = layers.Dense(
        units=self.edges * self.vertices * self.vertices, activation=None)
    self.edges_reshape = layers.Reshape((self.edges, self.vertices,
                                         self.vertices))
    self.edges_matrix_transpose1 = layers.Permute((1, 3, 2))
    self.edges_matrix_transpose2 = layers.Permute((2, 3, 1))
    self.edges_dropout = layers.Dropout(self.dropout_rate)

    # nodes logits used during training
    self.nodes_dense = layers.Dense(
        units=(self.vertices * self.nodes), activation=None)
    self.nodes_reshape = layers.Reshape((self.vertices, self.nodes))
    self.nodes_dropout = layers.Dropout(self.dropout_rate)

  def call(self,
           inputs: Any,
           training: bool = False,
           sample_generation: bool = False) -> List[Any]:
    """
    Call generator model

    Parameters
    ----------
    inputs : Any
        List of inputs, typically noise_batch
    training : bool, optional
        used by dropout layers, by default False
    sample_generation : bool, optional
        decide which output to use, by default False

    Returns
    -------
    List[Any, Any]
        Tensors containing either softmax values for training
        or argmax for sample generation (used for creation of rdkit molecules).
    """

    x = self.dense1(inputs)
    x = self.dropout1(x)
    x = self.dense2(x)
    x = self.dropout2(x)
    x = self.dense3(x)
    x = self.dropout3(x)

    # edges logits
    edges_logits = self.edges_dense(x)
    edges_logits = self.edges_reshape(edges_logits)
    matrix_transpose = self.edges_matrix_transpose1(edges_logits)
    edges_logits = (edges_logits + matrix_transpose) / 2
    edges_logits = self.edges_matrix_transpose2(edges_logits)
    edges_logits = self.edges_dropout(edges_logits)

    # nodes logits
    nodes_logits = self.nodes_dense(x)
    nodes_logits = self.nodes_reshape(nodes_logits)
    nodes_logits = self.nodes_dropout(nodes_logits)

    if sample_generation is False:
      # training of the model
      edges = tf.nn.softmax(edges_logits)
      nodes = tf.nn.softmax(nodes_logits)
    else:
      # generating compounds
      e_gumbel_logits = edges_logits - tf.math.log(-tf.math.log(
          tf.random.uniform(tf.shape(edges_logits), dtype=edges_logits.dtype)))
      e_gumbel_argmax = tf.one_hot(
          tf.argmax(e_gumbel_logits, axis=-1),
          depth=e_gumbel_logits.shape[-1],
          dtype=e_gumbel_logits.dtype,
      )
      edges = tf.argmax(e_gumbel_argmax, axis=-1)

      # nodes logits used during compound generation
      n_gumbel_logits = nodes_logits - tf.math.log(-tf.math.log(
          tf.random.uniform(tf.shape(nodes_logits), dtype=nodes_logits.dtype)))
      n_gumbel_argmax = tf.one_hot(
          tf.argmax(n_gumbel_logits, axis=-1),
          depth=n_gumbel_logits.shape[-1],
          dtype=n_gumbel_logits.dtype,
      )
      nodes = tf.argmax(n_gumbel_argmax, axis=-1)

    return [edges, nodes]
