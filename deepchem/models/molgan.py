import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from deepchem.models import WGAN
from deepchem.models.layers import MolGANEncoderLayer
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix


class BasicMolGANModel(WGAN):
  """Model for automatic generation of compounds based on GAN architecture described by Nicola De Cao et al.
    `MolGAN: An implicit generative model for small molecular graphs`<https://arxiv.org/abs/1805.11973>`_.
    It uses adjacency matrix and node features as inputs, both need to be converted to one hot representation before use.


    Examples
    --------
    gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000))
    dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in limited],[x.node_features for x in limited])
    def iterbatches(epochs):
        for i in range(epochs):
            for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
                adjacency_tensor = tf.one_hot(batch[0], gan.edges)
                node_tesor = tf.one_hot(batch[1], gan.nodes)
                yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tesor}
    gan.fit_gan(iterbatches(10), generator_steps=0.2, checkpoint_interval=5000)

    """

  def __init__(self,
               edges: int = 5,
               vertices: int = 9,
               nodes: int = 5,
               embedding_dim: int = 10,
               dropout_rate: float = 0.0,
               name: str = "",
               **kwargs):
    """
        Parameters
        ----------
        edges: int, default 5
            Number of bond types includes BondType.Zero
        vertices: int, default 9
            Max number of atoms in adjacency and node features matrices
        nodes: int, default 5
            Number of atom types in node features matrix
        embedding_dim: int, default 10
            Size of noise input
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

    super(BasicMolGAN, self).__init__(name=name, **kwargs)

  def get_noise_input_shape(self):
    return (self.embedding_dim,)

  def get_data_input_shapes(self):
    return [(self.vertices, self.vertices, self.edges), (self.vertices,
                                                         self.nodes)]

  def create_generator(self):
    input_layer = layers.Input(shape=(self.embedding_dim,))
    x = layers.Dense(128, activation="tanh")(input_layer)
    x = layers.Dropout(self.dropout_rate)(x)
    x = layers.Dense(256, activation="tanh")(x)
    x = layers.Dropout(self.dropout_rate)(x)
    x = layers.Dense(512, activation="tanh")(x)
    x = layers.Dropout(self.dropout_rate)(x)

    # EDGES LOGITS
    edges_logits = layers.Dense(
        units=self.edges * self.vertices * self.vertices, activation=None)(x)
    edges_logits = layers.Reshape((self.edges, self.vertices,
                                   self.vertices))(edges_logits)
    matrix_transpose = layers.Permute((1, 3, 2))(edges_logits)
    edges_logits = (edges_logits + matrix_transpose) / 2
    edges_logits = layers.Permute((2, 3, 1))(edges_logits)
    edges_logits = layers.Dropout(self.dropout_rate)(edges_logits)

    # used during training of the model
    edges_softmax = tf.nn.softmax(edges_logits)

    # NODES LOGITS
    nodes_logits = layers.Dense(
        units=(self.vertices * self.nodes), activation=None)(x)
    nodes_logits = layers.Reshape((self.vertices, self.nodes))(nodes_logits)
    nodes_logits = layers.Dropout(self.dropout_rate)(nodes_logits)

    # used during training of the model
    nodes_softmax = tf.nn.softmax(nodes_logits)

    # used to generate molecules, consider returning just logits and then use additonal layer when mols needs to generated

    # used for compound generation, consider removing this from this section and just return un
    e_gumbel_logits = edges_logits - tf.math.log(-tf.math.log(
        tf.random.uniform(tf.shape(edges_logits), dtype=edges_logits.dtype)))
    e_gumbel_argmax = tf.one_hot(
        tf.argmax(e_gumbel_logits, axis=-1),
        depth=e_gumbel_logits.shape[-1],
        dtype=e_gumbel_logits.dtype,
    )
    e_argmax = tf.argmax(e_gumbel_argmax, axis=-1)

    # used for compound generation
    n_gumbel_logits = nodes_logits - tf.math.log(-tf.math.log(
        tf.random.uniform(tf.shape(nodes_logits), dtype=nodes_logits.dtype)))
    n_gumbel_argmax = tf.one_hot(
        tf.argmax(n_gumbel_logits, axis=-1),
        depth=n_gumbel_logits.shape[-1],
        dtype=n_gumbel_logits.dtype,
    )
    n_argmax = tf.argmax(n_gumbel_argmax, axis=-1)

    # final model
    return keras.Model(
        inputs=input_layer,
        outputs=[edges_softmax, nodes_softmax, e_argmax, n_argmax],
    )

  def create_discriminator(self):
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

    return keras.Model(inputs=[adjacency_tensor, node_tensor], outputs=[output])

  def predict_gan_generator(self,
                            batch_size=1,
                            noise_input=None,
                            generator_index=0):
    """Use the GAN to generate a batch of samples.
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
        generator_index: int
          the index of the generator (between 0 and n_generators-1) to use for
          generating the samples.
        Returns
        -------
        An array (if the generator has only one output) or list of arrays (if it has
        multiple outputs) containing the generated samples.
        """
    if noise_input is not None:
      batch_size = len(noise_input)
    if noise_input is None:
      noise_input = self.get_noise_batch(batch_size)
    inputs = noise_input
    _, _, adjacency_matrix, nodes_features = self.generators[0](
        inputs, training=False)
    graphs = [
        GraphMatrix(i, j)
        for i, j in zip(adjacency_matrix.numpy(), nodes_features.numpy())
    ]
    return graphs
