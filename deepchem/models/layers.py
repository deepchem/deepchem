import tensorflow as tf
import numpy as np
import collections
from deepchem.models.tensorgraph import model_ops, initializations, activations

class InteratomicL2Distances(tf.keras.layers.Layer):
  """Compute (squared) L2 Distances between atoms given neighbors."""

  def __init__(self, N_atoms, M_nbrs, ndim, **kwargs):
    super(InteratomicL2Distances, self).__init__(**kwargs)
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim

  def call(self, inputs):
    if len(inputs) != 2:
      raise ValueError("InteratomicDistances requires coords,nbr_list")
    coords, nbr_list = (inputs[0], inputs[1])
    N_atoms, M_nbrs, ndim = self.N_atoms, self.M_nbrs, self.ndim
    # Shape (N_atoms, M_nbrs, ndim)
    nbr_coords = tf.gather(coords, nbr_list)
    # Shape (N_atoms, M_nbrs, ndim)
    tiled_coords = tf.tile(
        tf.reshape(coords, (N_atoms, 1, ndim)), (1, M_nbrs, 1))
    # Shape (N_atoms, M_nbrs)
    return tf.reduce_sum((tiled_coords - nbr_coords)**2, axis=2)

class GraphConv(tf.keras.layers.Layer):

  def __init__(self,
               out_channel,
               min_deg=0,
               max_deg=10,
               activation_fn=None,
               **kwargs):
    super(GraphConv, self).__init__(**kwargs)
    self.out_channel = out_channel
    self.min_degree = min_deg
    self.max_degree = max_deg
    self.activation_fn = activation_fn

  def build(self, input_shape):
    # Generate the nb_affine weights and biases
    num_deg = 2 * self.max_degree + (1 - self.min_degree)
    self.W_list = [
        self.add_weight(name='kernel', shape=(int(input_shape[0][-1]), self.out_channel), initializer='glorot_uniform', trainable=True)
        for k in range(num_deg)
    ]
    self.b_list = [
        self.add_weight(name='bias', shape=(self.out_channel,), initializer='zeros', trainable=True)
        for k in range(num_deg)
    ]

  def call(self, inputs):

    # Extract atom_features
    atom_features = inputs[0]

    # Extract graph topology
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    W = iter(self.W_list)
    b = iter(self.b_list)

    # Sum all neighbors using adjacency matrix
    deg_summed = self.sum_neigh(atom_features, deg_adj_lists)

    # Get collection of modified atom features
    new_rel_atoms_collection = (self.max_degree + 1 - self.min_degree) * [None]

    for deg in range(1, self.max_degree + 1):
      # Obtain relevant atoms for this degree
      rel_atoms = deg_summed[deg - 1]

      # Get self atoms
      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Apply hidden affine to relevant atoms and append
      rel_out = tf.matmul(rel_atoms, next(W)) + next(b)
      self_out = tf.matmul(self_atoms, next(W)) + next(b)
      out = rel_out + self_out

      new_rel_atoms_collection[deg - self.min_degree] = out

    # Determine the min_deg=0 case
    if self.min_degree == 0:
      deg = 0

      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Only use the self layer
      out = tf.matmul(self_atoms, next(W)) + next(b)

      new_rel_atoms_collection[deg - self.min_degree] = out

    # Combine all atoms back into the list
    atom_features = tf.concat(axis=0, values=new_rel_atoms_collection)

    if self.activation_fn is not None:
      atom_features = self.activation_fn(atom_features)

    return atom_features

  def sum_neigh(self, atoms, deg_adj_lists):
    """Store the summed atoms by degree"""
    deg_summed = self.max_degree * [None]

    # Tensorflow correctly processes empty lists when using concat
    for deg in range(1, self.max_degree + 1):
      gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
      # Sum along neighbors as well as self, and store
      summed_atoms = tf.reduce_sum(gathered_atoms, 1)
      deg_summed[deg - 1] = summed_atoms

    return deg_summed


class GraphPool(tf.keras.layers.Layer):

  def __init__(self, min_degree=0, max_degree=10, **kwargs):
    super(GraphPool, self).__init__(**kwargs)
    self.min_degree = min_degree
    self.max_degree = max_degree

  def call(self, inputs):
    atom_features = inputs[0]
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    # Perform the mol gather
    # atom_features = graph_pool(atom_features, deg_adj_lists, deg_slice,
    #                            self.max_degree, self.min_degree)

    deg_maxed = (self.max_degree + 1 - self.min_degree) * [None]

    # Tensorflow correctly processes empty lists when using concat

    for deg in range(1, self.max_degree + 1):
      # Get self atoms
      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Expand dims
      self_atoms = tf.expand_dims(self_atoms, 1)

      # always deg-1 for deg_adj_lists
      gathered_atoms = tf.gather(atom_features, deg_adj_lists[deg - 1])
      gathered_atoms = tf.concat(axis=1, values=[self_atoms, gathered_atoms])

      maxed_atoms = tf.reduce_max(gathered_atoms, 1)
      deg_maxed[deg - self.min_degree] = maxed_atoms

    if self.min_degree == 0:
      begin = tf.stack([deg_slice[0, 0], 0])
      size = tf.stack([deg_slice[0, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)
      deg_maxed[0] = self_atoms

    return tf.concat(axis=0, values=deg_maxed)


class GraphGather(tf.keras.layers.Layer):

  def __init__(self, batch_size, activation_fn=None, **kwargs):
    super(GraphGather, self).__init__(**kwargs)
    self.batch_size = batch_size
    self.activation_fn = activation_fn

  def call(self, inputs):
    # x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]
    atom_features = inputs[0]

    # Extract graph topology
    membership = inputs[2]

    assert self.batch_size > 1, "graph_gather requires batches larger than 1"

    sparse_reps = tf.unsorted_segment_sum(atom_features, membership,
                                          self.batch_size)
    max_reps = tf.unsorted_segment_max(atom_features, membership,
                                       self.batch_size)
    mol_features = tf.concat(axis=1, values=[sparse_reps, max_reps])

    if self.activation_fn is not None:
      mol_features = self.activation_fn(mol_features)
    return mol_features


class LSTMStep(tf.keras.layers.Layer):
  """Layer that performs a single step LSTM update.

  This layer performs a single step LSTM update. Note that it is *not*
  a full LSTM recurrent network. The LSTMStep layer is useful as a
  primitive for designing layers such as the AttnLSTMEmbedding or the
  IterRefLSTMEmbedding below.
  """

  def __init__(self,
               output_dim,
               input_dim,
               init_fn=initializations.glorot_uniform,
               inner_init_fn=initializations.orthogonal,
               activation_fn=activations.tanh,
               inner_activation_fn=activations.hard_sigmoid,
               **kwargs):
    """
    Parameters
    ----------
    output_dim: int
      Dimensionality of output vectors.
    input_dim: int
      Dimensionality of input vectors.
    init_fn: object
      TensorFlow initialization to use for W.
    inner_init_fn: object
      TensorFlow initialization to use for U.
    activation_fn: object
      TensorFlow activation to use for output.
    inner_activation_fn: object
      TensorFlow activation to use for inner steps.
    """

    super(LSTMStep, self).__init__(**kwargs)
    self.init = init_fn
    self.inner_init = inner_init_fn
    self.output_dim = output_dim
    # No other forget biases supported right now.
    self.activation = activation_fn
    self.inner_activation = inner_activation_fn
    self.input_dim = input_dim

  def get_initial_states(self, input_shape):
    return [model_ops.zeros(input_shape), model_ops.zeros(input_shape)]

  def build(self, input_shape):
    """Constructs learnable weights for this layer."""
    init = self.init
    inner_init = self.inner_init
    self.W = init((self.input_dim, 4 * self.output_dim))
    self.U = inner_init((self.output_dim, 4 * self.output_dim))

    self.b = tf.Variable(
        np.hstack((np.zeros(self.output_dim), np.ones(self.output_dim),
                   np.zeros(self.output_dim), np.zeros(self.output_dim))),
        dtype=tf.float32)

  def call(self, inputs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    inputs: list
      List of three tensors (x, h_tm1, c_tm1). h_tm1 means "h, t-1".

    Returns
    -------
    list
      Returns h, [h, c]
    """
    activation = self.activation
    inner_activation = self.inner_activation
    x, h_tm1, c_tm1 = inputs

    # Taken from Keras code [citation needed]
    z = model_ops.dot(x, self.W) + model_ops.dot(h_tm1, self.U) + self.b

    z0 = z[:, :self.output_dim]
    z1 = z[:, self.output_dim:2 * self.output_dim]
    z2 = z[:, 2 * self.output_dim:3 * self.output_dim]
    z3 = z[:, 3 * self.output_dim:]

    i = inner_activation(z0)
    f = inner_activation(z1)
    c = f * c_tm1 + i * activation(z2)
    o = inner_activation(z3)

    h = o * activation(c)
    return h, [h, c]


def _cosine_dist(x, y):
  """Computes the inner product (cosine distance) between two tensors.

  Parameters
  ----------
  x: tf.Tensor
    Input Tensor
  y: tf.Tensor
    Input Tensor
  """
  denom = (
      model_ops.sqrt(model_ops.sum(tf.square(x)) * model_ops.sum(tf.square(y)))
      + model_ops.epsilon())
  return model_ops.dot(x, tf.transpose(y)) / denom


class AttnLSTMEmbedding(tf.keras.layers.Layer):
  """Implements AttnLSTM as in matching networks paper.

  The AttnLSTM embedding adjusts two sets of vectors, the "test" and
  "support" sets. The "support" consists of a set of evidence vectors.
  Think of these as the small training set for low-data machine
  learning.  The "test" consists of the queries we wish to answer with
  the small amounts ofavailable data. The AttnLSTMEmbdding allows us to
  modify the embedding of the "test" set depending on the contents of
  the "support".  The AttnLSTMEmbedding is thus a type of learnable
  metric that allows a network to modify its internal notion of
  distance.

  References:
  Matching Networks for One Shot Learning
  https://arxiv.org/pdf/1606.04080v1.pdf

  Order Matters: Sequence to sequence for sets
  https://arxiv.org/abs/1511.06391
  """

  def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
    """
    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of features per atom
    max_depth: int
      Number of "processing steps" used by sequence-to-sequence for sets model.
    """
    super(AttnLSTMEmbedding, self).__init__(**kwargs)

    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  def build(self, input_shape):
    n_feat = self.n_feat
    self.lstm = LSTMStep(n_feat, 2 * n_feat)
    self.q_init = model_ops.zeros([self.n_test, n_feat])
    self.states_init = self.lstm.get_initial_states([self.n_test, n_feat])

  def call(self, inputs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    inputs: list
      List of two tensors (X, Xp). X should be of shape (n_test,
      n_feat) and Xp should be of shape (n_support, n_feat) where
      n_test is the size of the test set, n_support that of the support
      set, and n_feat is the number of per-atom features.

    Returns
    -------
    list
      Returns two tensors of same shape as input. Namely the output
      shape will be [(n_test, n_feat), (n_support, n_feat)]
    """
    if len(inputs) != 2:
      raise ValueError("AttnLSTMEmbedding layer must have exactly two parents")
    # x is test set, xp is support set.
    x, xp = inputs

    # Get initializations
    q = self.q_init
    states = self.states_init

    for d in range(self.max_depth):
      # Process using attention
      # Eqn (4), appendix A.1 of Matching Networks paper
      e = _cosine_dist(x + q, xp)
      a = tf.nn.softmax(e)
      r = model_ops.dot(a, xp)

      # Generate new attention states
      y = model_ops.concatenate([q, r], axis=1)
      q, states = self.lstm([y]+states)
    return [x + q, xp]


class IterRefLSTMEmbedding(tf.keras.layers.Layer):
  """Implements the Iterative Refinement LSTM.

  Much like AttnLSTMEmbedding, the IterRefLSTMEmbedding is another type
  of learnable metric which adjusts "test" and "support." Recall that
  "support" is the small amount of data available in a low data machine
  learning problem, and that "test" is the query. The AttnLSTMEmbedding
  only modifies the "test" based on the contents of the support.
  However, the IterRefLSTM modifies both the "support" and "test" based
  on each other. This allows the learnable metric to be more malleable
  than that from AttnLSTMEmbeding.
  """

  def __init__(self, n_test, n_support, n_feat, max_depth, **kwargs):
    """
    Unlike the AttnLSTM model which only modifies the test vectors
    additively, this model allows for an additive update to be
    performed to both test and support using information from each
    other.

    Parameters
    ----------
    n_support: int
      Size of support set.
    n_test: int
      Size of test set.
    n_feat: int
      Number of input atom features
    max_depth: int
      Number of LSTM Embedding layers.
    """
    super(IterRefLSTMEmbedding, self).__init__(**kwargs)
    self.max_depth = max_depth
    self.n_test = n_test
    self.n_support = n_support
    self.n_feat = n_feat

  def build(self, input_shape):
    n_feat = self.n_feat

    # Support set lstm
    self.support_lstm = LSTMStep(n_feat, 2 * n_feat)
    self.q_init = model_ops.zeros([self.n_support, n_feat])
    self.support_states_init = self.support_lstm.get_initial_states(
        [self.n_support, n_feat])

    # Test lstm
    self.test_lstm = LSTMStep(n_feat, 2 * n_feat)
    self.p_init = model_ops.zeros([self.n_test, n_feat])
    self.test_states_init = self.test_lstm.get_initial_states([self.n_test, n_feat])

  def call(self, inputs):
    """Execute this layer on input tensors.

    Parameters
    ----------
    inputs: list
      List of two tensors (X, Xp). X should be of shape (n_test, n_feat) and
      Xp should be of shape (n_support, n_feat) where n_test is the size of
      the test set, n_support that of the support set, and n_feat is the number
      of per-atom features.

    Returns
    -------
    list
      Returns two tensors of same shape as input. Namely the output shape will
      be [(n_test, n_feat), (n_support, n_feat)]
    """
    if len(inputs) != 2:
      raise ValueError(
          "IterRefLSTMEmbedding layer must have exactly two parents")
    x, xp = inputs

    # Get initializations
    p = self.p_init
    q = self.q_init
    # Rename support
    z = xp
    states = self.support_states_init
    x_states = self.test_states_init

    for d in range(self.max_depth):
      # Process support xp using attention
      e = _cosine_dist(z + q, xp)
      a = tf.nn.softmax(e)
      # Get linear combination of support set
      r = model_ops.dot(a, xp)

      # Process test x using attention
      x_e = _cosine_dist(x + p, z)
      x_a = tf.nn.softmax(x_e)
      s = model_ops.dot(x_a, z)

      # Generate new support attention states
      qr = model_ops.concatenate([q, r], axis=1)
      q, states = self.support_lstm([qr]+states)

      # Generate new test attention states
      ps = model_ops.concatenate([p, s], axis=1)
      p, x_states = self.test_lstm([ps]+x_states)

      # Redefine
      z = r

    return [x + p, xp + q]


class WeightedLinearCombo(tf.keras.layers.Layer):
  """Computes a weighted linear combination of input layers, with the weights defined by trainable variables."""

  def __init__(self, std=0.3, **kwargs):
    super(WeightedLinearCombo, self).__init__(**kwargs)
    self.std = std

  def build(self, input_shape):
    init = tf.keras.initializers.RandomNormal(stddev=self.std)
    self.input_weights = [
        self.add_weight('weight_%d' % (i+1), (1,), initializer=init, trainable=True)
        for i in range(len(input_shape))
    ]

  def call(self, inputs):
    out_tensor = None
    for in_tensor, w in zip(inputs, self.input_weights):
      if out_tensor is None:
        out_tensor = w * in_tensor
      else:
        out_tensor += w * in_tensor
    return out_tensor


class VinaFreeEnergy(tf.keras.layers.Layer):
  """Computes free-energy as defined by Autodock Vina.

  TODO(rbharath): Make this layer support batching.
  """

  def __init__(self,
               N_atoms,
               M_nbrs,
               ndim,
               nbr_cutoff,
               start,
               stop,
               stddev=.3,
               Nrot=1,
               **kwargs):
    super(VinaFreeEnergy, self).__init__(**kwargs)
    self.stddev = stddev
    # Number of rotatable bonds
    # TODO(rbharath): Vina actually sets this per-molecule. See if makes
    # a difference.
    self.Nrot = Nrot
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim
    self.nbr_cutoff = nbr_cutoff
    self.start = start
    self.stop = stop

  def build(self, input_shape):
    self.weighted_combo = WeightedLinearCombo()
    self.w = tf.Variable(tf.random_normal((1,), stddev=self.stddev))

  def cutoff(self, d, x):
    out_tensor = tf.where(d < 8, x, tf.zeros_like(x))
    return out_tensor

  def nonlinearity(self, c, w):
    """Computes non-linearity used in Vina."""
    out_tensor = c / (1 + w * self.Nrot)
    return w, out_tensor

  def repulsion(self, d):
    """Computes Autodock Vina's repulsion interaction term."""
    out_tensor = tf.where(d < 0, d**2, tf.zeros_like(d))
    return out_tensor

  def hydrophobic(self, d):
    """Computes Autodock Vina's hydrophobic interaction term."""
    out_tensor = tf.where(d < 0.5, tf.ones_like(d),
                          tf.where(d < 1.5, 1.5 - d, tf.zeros_like(d)))
    return out_tensor

  def hydrogen_bond(self, d):
    """Computes Autodock Vina's hydrogen bond interaction term."""
    out_tensor = tf.where(
        d < -0.7, tf.ones_like(d),
        tf.where(d < 0, (1.0 / 0.7) * (0 - d), tf.zeros_like(d)))
    return out_tensor

  def gaussian_first(self, d):
    """Computes Autodock Vina's first Gaussian interaction term."""
    out_tensor = tf.exp(-(d / 0.5)**2)
    return out_tensor

  def gaussian_second(self, d):
    """Computes Autodock Vina's second Gaussian interaction term."""
    out_tensor = tf.exp(-((d - 3) / 2)**2)
    return out_tensor

  def call(self, inputs):
    """
    Parameters
    ----------
    X: tf.Tensor of shape (N, d)
      Coordinates/features.
    Z: tf.Tensor of shape (N)
      Atomic numbers of neighbor atoms.

    Returns
    -------
    layer: tf.Tensor of shape (B)
      The free energy of each complex in batch
    """
    X = inputs[0]
    Z = inputs[1]

    # TODO(rbharath): This layer shouldn't be neighbor-listing. Make
    # neighbors lists an argument instead of a part of this layer.
    nbr_list = NeighborList(self.N_atoms, self.M_nbrs, self.ndim,
                            self.nbr_cutoff, self.start, self.stop)(X)

    # Shape (N, M)
    dists = InteratomicL2Distances(self.N_atoms, self.M_nbrs,
                                   self.ndim)([X, nbr_list])

    repulsion = self.repulsion(dists)
    hydrophobic = self.hydrophobic(dists)
    hbond = self.hydrogen_bond(dists)
    gauss_1 = self.gaussian_first(dists)
    gauss_2 = self.gaussian_second(dists)

    # Shape (N, M)
    interactions = self.weighted_combo([repulsion, hydrophobic, hbond, gauss_1,
                                  gauss_2])

    # Shape (N, M)
    thresholded = self.cutoff(dists, interactions)

    weight, free_energies = self.nonlinearity(thresholded, self.w)
    return tf.reduce_sum(free_energies)


class NeighborList(tf.keras.layers.Layer):
  """Computes a neighbor-list in Tensorflow.

  Neighbor-lists (also called Verlet Lists) are a tool for grouping atoms which
  are close to each other spatially

  TODO(rbharath): Make this layer support batching.
  """

  def __init__(self, N_atoms, M_nbrs, ndim, nbr_cutoff, start, stop, **kwargs):
    """
    Parameters
    ----------
    N_atoms: int
      Maximum number of atoms this layer will neighbor-list.
    M_nbrs: int
      Maximum number of spatial neighbors possible for atom.
    ndim: int
      Dimensionality of space atoms live in. (Typically 3D, but sometimes will
      want to use higher dimensional descriptors for atoms).
    nbr_cutoff: float
      Length in Angstroms (?) at which atom boxes are gridded.
    """
    super(NeighborList, self).__init__(**kwargs)
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim
    # Number of grid cells
    n_cells = int(((stop - start) / nbr_cutoff)**ndim)
    self.n_cells = n_cells
    self.nbr_cutoff = nbr_cutoff
    self.start = start
    self.stop = stop

  def call(self, inputs):
    if isinstance(inputs, collections.Sequence):
      if len(inputs) != 1:
        raise ValueError("NeighborList can only have one input")
      inputs = inputs[0]
    if len(inputs.get_shape()) != 2:
      # TODO(rbharath): Support batching
      raise ValueError("Parent tensor must be (num_atoms, ndum)")
    return self.compute_nbr_list(inputs)

  def compute_nbr_list(self, coords):
    """Get closest neighbors for atoms.

    Needs to handle padding for atoms with no neighbors.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N_atoms, ndim)

    Returns
    -------
    nbr_list: tf.Tensor
      Shape (N_atoms, M_nbrs) of atom indices
    """
    # Shape (n_cells, ndim)
    cells = self.get_cells()

    # List of length N_atoms, each element of different length uniques_i
    nbrs = self.get_atoms_in_nbrs(coords, cells)
    padding = tf.fill((self.M_nbrs,), -1)
    padded_nbrs = [tf.concat([unique_nbrs, padding], 0) for unique_nbrs in nbrs]

    # List of length N_atoms, each element of different length uniques_i
    # List of length N_atoms, each a tensor of shape
    # (uniques_i, ndim)
    nbr_coords = [tf.gather(coords, atom_nbrs) for atom_nbrs in nbrs]

    # Add phantom atoms that exist far outside the box
    coord_padding = tf.cast(
        tf.fill((self.M_nbrs, self.ndim), 2 * self.stop), tf.float32)
    padded_nbr_coords = [
        tf.concat([nbr_coord, coord_padding], 0) for nbr_coord in nbr_coords
    ]

    # List of length N_atoms, each of shape (1, ndim)
    atom_coords = tf.split(coords, self.N_atoms)
    # TODO(rbharath): How does distance need to be modified here to
    # account for periodic boundary conditions?
    # List of length N_atoms each of shape (M_nbrs)
    padded_dists = [
        tf.reduce_sum((atom_coord - padded_nbr_coord)**2, axis=1)
        for (atom_coord,
             padded_nbr_coord) in zip(atom_coords, padded_nbr_coords)
    ]

    padded_closest_nbrs = [
        tf.nn.top_k(-padded_dist, k=self.M_nbrs)[1]
        for padded_dist in padded_dists
    ]

    # N_atoms elts of size (M_nbrs,) each
    padded_neighbor_list = [
        tf.gather(padded_atom_nbrs, padded_closest_nbr)
        for (padded_atom_nbrs,
             padded_closest_nbr) in zip(padded_nbrs, padded_closest_nbrs)
    ]

    neighbor_list = tf.stack(padded_neighbor_list)

    return neighbor_list

  def get_atoms_in_nbrs(self, coords, cells):
    """Get the atoms in neighboring cells for each cells.

    Returns
    -------
    atoms_in_nbrs = (N_atoms, n_nbr_cells, M_nbrs)
    """
    # Shape (N_atoms, 1)
    cells_for_atoms = self.get_cells_for_atoms(coords, cells)

    # Find M_nbrs atoms closest to each cell
    # Shape (n_cells, M_nbrs)
    closest_atoms = self.get_closest_atoms(coords, cells)

    # Associate each cell with its neighbor cells. Assumes periodic boundary
    # conditions, so does wrapround. O(constant)
    # Shape (n_cells, n_nbr_cells)
    neighbor_cells = self.get_neighbor_cells(cells)

    # Shape (N_atoms, n_nbr_cells)
    neighbor_cells = tf.squeeze(tf.gather(neighbor_cells, cells_for_atoms))

    # Shape (N_atoms, n_nbr_cells, M_nbrs)
    atoms_in_nbrs = tf.gather(closest_atoms, neighbor_cells)

    # Shape (N_atoms, n_nbr_cells*M_nbrs)
    atoms_in_nbrs = tf.reshape(atoms_in_nbrs, [self.N_atoms, -1])

    # List of length N_atoms, each element length uniques_i
    nbrs_per_atom = tf.split(atoms_in_nbrs, self.N_atoms)
    uniques = [
        tf.unique(tf.squeeze(atom_nbrs))[0] for atom_nbrs in nbrs_per_atom
    ]

    # TODO(rbharath): FRAGILE! Uses fact that identity seems to be the first
    # element removed to remove self from list of neighbors. Need to verify
    # this holds more broadly or come up with robust alternative.
    uniques = [unique[1:] for unique in uniques]

    return uniques

  def get_closest_atoms(self, coords, cells):
    """For each cell, find M_nbrs closest atoms.

    Let N_atoms be the number of atoms.

    Parameters
    ----------
    coords: tf.Tensor
      (N_atoms, ndim) shape.
    cells: tf.Tensor
      (n_cells, ndim) shape.

    Returns
    -------
    closest_inds: tf.Tensor
      Of shape (n_cells, M_nbrs)
    """
    N_atoms, n_cells, ndim, M_nbrs = (self.N_atoms, self.n_cells, self.ndim,
                                      self.M_nbrs)
    # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
    tiled_cells = tf.reshape(
        tf.tile(cells, (1, N_atoms)), (N_atoms * n_cells, ndim))

    # Shape (N_atoms*n_cells, ndim) after tile
    tiled_coords = tf.tile(coords, (n_cells, 1))

    # Shape (N_atoms*n_cells)
    coords_vec = tf.reduce_sum((tiled_coords - tiled_cells)**2, axis=1)
    # Shape (n_cells, N_atoms)
    coords_norm = tf.reshape(coords_vec, (n_cells, N_atoms))

    # Find k atoms closest to this cell. Notice negative sign since
    # tf.nn.top_k returns *largest* not smallest.
    # Tensor of shape (n_cells, M_nbrs)
    closest_inds = tf.nn.top_k(-coords_norm, k=M_nbrs)[1]

    return closest_inds

  def get_cells_for_atoms(self, coords, cells):
    """Compute the cells each atom belongs to.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N_atoms, ndim)
    cells: tf.Tensor
      (n_cells, ndim) shape.
    Returns
    -------
    cells_for_atoms: tf.Tensor
      Shape (N_atoms, 1)
    """
    N_atoms, n_cells, ndim = self.N_atoms, self.n_cells, self.ndim
    n_cells = int(n_cells)
    # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
    tiled_cells = tf.tile(cells, (N_atoms, 1))

    # Shape (N_atoms*n_cells, 1) after tile
    tiled_coords = tf.reshape(
        tf.tile(coords, (1, n_cells)), (n_cells * N_atoms, ndim))
    coords_vec = tf.reduce_sum((tiled_coords - tiled_cells)**2, axis=1)
    coords_norm = tf.reshape(coords_vec, (N_atoms, n_cells))

    closest_inds = tf.nn.top_k(-coords_norm, k=1)[1]
    return closest_inds

  def _get_num_nbrs(self):
    """Get number of neighbors in current dimensionality space."""
    ndim = self.ndim
    if ndim == 1:
      n_nbr_cells = 3
    elif ndim == 2:
      # 9 neighbors in 2-space
      n_nbr_cells = 9
    # TODO(rbharath): Shoddy handling of higher dimensions...
    elif ndim >= 3:
      # Number of cells for cube in 3-space is
      n_nbr_cells = 27  # (26 faces on Rubik's cube for example)
    return n_nbr_cells

  def get_neighbor_cells(self, cells):
    """Compute neighbors of cells in grid.

    # TODO(rbharath): Do we need to handle periodic boundary conditions
    properly here?
    # TODO(rbharath): This doesn't handle boundaries well. We hard-code
    # looking for n_nbr_cells neighbors, which isn't right for boundary cells in
    # the cube.

    Parameters
    ----------
    cells: tf.Tensor
      (n_cells, ndim) shape.
    Returns
    -------
    nbr_cells: tf.Tensor
      (n_cells, n_nbr_cells)
    """
    ndim, n_cells = self.ndim, self.n_cells
    n_nbr_cells = self._get_num_nbrs()
    # Tile cells to form arrays of size (n_cells*n_cells, ndim)
    # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
    # Tile (a, a, a, b, b, b, etc.)
    tiled_centers = tf.reshape(
        tf.tile(cells, (1, n_cells)), (n_cells * n_cells, ndim))
    # Tile (a, b, c, a, b, c, ...)
    tiled_cells = tf.tile(cells, (n_cells, 1))

    coords_vec = tf.reduce_sum((tiled_centers - tiled_cells)**2, axis=1)
    coords_norm = tf.reshape(coords_vec, (n_cells, n_cells))
    closest_inds = tf.nn.top_k(-coords_norm, k=n_nbr_cells)[1]

    return closest_inds

  def get_cells(self):
    """Returns the locations of all grid points in box.

    Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
    Then would return a list of length 20^3 whose entries would be
    [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

    Returns
    -------
    cells: tf.Tensor
      (n_cells, ndim) shape.
    """
    start, stop, nbr_cutoff = self.start, self.stop, self.nbr_cutoff
    mesh_args = [tf.range(start, stop, nbr_cutoff) for _ in range(self.ndim)]
    return tf.cast(
        tf.reshape(
            tf.transpose(tf.stack(tf.meshgrid(*mesh_args))),
            (self.n_cells, self.ndim)), tf.float32)


class AtomicConvolution(tf.keras.layers.Layer):

  def __init__(self,
               atom_types=None,
               radial_params=list(),
               boxsize=None,
               **kwargs):
    """Atomic convoluation layer

    N = max_num_atoms, M = max_num_neighbors, B = batch_size, d = num_features
    l = num_radial_filters * num_atom_types

    Parameters
    ----------
    atom_types: list or None
      Of length a, where a is number of atom types for filtering.
    radial_params: list
      Of length l, where l is number of radial filters learned.
    boxsize: float or None
      Simulation box length [Angstrom].
    """
    super(AtomicConvolution, self).__init__(**kwargs)
    self.boxsize = boxsize
    self.radial_params = radial_params
    self.atom_types = atom_types

  def build(self, input_shape):
    vars = []
    for i in range(3):
      val = np.array([p[i] for p in self.radial_params]).reshape((-1, 1, 1, 1))
      vars.append(tf.Variable(val, dtype=tf.float32))
    self.rc = vars[0]
    self.rs = vars[1]
    self.re = vars[2]

  def call(self, inputs):
    """
    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features.
    Nbrs: tf.Tensor of shape (B, N, M)
      Neighbor list.
    Nbrs_Z: tf.Tensor of shape (B, N, M)
      Atomic numbers of neighbor atoms.

    Returns
    -------
    layer: tf.Tensor of shape (B, N, l)
      A new tensor representing the output of the atomic conv layer
    """
    X = inputs[0]
    Nbrs = tf.cast(inputs[1], tf.int32)
    Nbrs_Z = inputs[2]

    # N: Maximum number of atoms
    # M: Maximum number of neighbors
    # d: Number of coordinates/features/filters
    # B: Batch Size
    N = X.get_shape()[-2].value
    d = X.get_shape()[-1].value
    M = Nbrs.get_shape()[-1].value
    B = X.get_shape()[0].value

    # Compute the distances and radial symmetry functions.
    D = self.distance_tensor(X, Nbrs, self.boxsize, B, N, M, d)
    R = self.distance_matrix(D)
    R = tf.reshape(R, [1] + R.shape.as_list())
    rsf = self.radial_symmetry_function(R, self.rc, self.rs, self.re)

    if not self.atom_types:
      cond = tf.cast(tf.not_equal(Nbrs_Z, 0), tf.float32)
      cond = tf.reshape(cond, R.shape)
      layer = tf.reduce_sum(cond * rsf, 3)
    else:
      sym = []
      for j in range(len(self.atom_types)):
        cond = tf.cast(tf.equal(Nbrs_Z, self.atom_types[j]), tf.float32)
        cond = tf.reshape(cond, R.shape)
        sym.append(tf.reduce_sum(cond * rsf, 3))
      layer = tf.concat(sym, 0)

    layer = tf.transpose(layer, [1, 2, 0])  # (l, B, N) -> (B, N, l)
    m, v = tf.nn.moments(layer, axes=[0])
    return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

  def radial_symmetry_function(self, R, rc, rs, e):
    """Calculates radial symmetry function.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_filters

    Parameters
    ----------
    R: tf.Tensor of shape (B, N, M)
      Distance matrix.
    rc: float
      Interaction cutoff [Angstrom].
    rs: float
      Gaussian distance matrix mean.
    e: float
      Gaussian distance matrix width.

    Returns
    -------
    retval: tf.Tensor of shape (B, N, M)
      Radial symmetry function (before summation)
    """
    K = self.gaussian_distance_matrix(R, rs, e)
    FC = self.radial_cutoff(R, rc)
    return tf.multiply(K, FC)

  def radial_cutoff(self, R, rc):
    """Calculates radial cutoff matrix.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors

    Parameters
    ----------
      R [B, N, M]: tf.Tensor
        Distance matrix.
      rc: tf.Variable
        Interaction cutoff [Angstrom].

    Returns
    -------
      FC [B, N, M]: tf.Tensor
        Radial cutoff matrix.
    """
    T = 0.5 * (tf.cos(np.pi * R / (rc)) + 1)
    E = tf.zeros_like(T)
    cond = tf.less_equal(R, rc)
    FC = tf.where(cond, T, E)
    return FC

  def gaussian_distance_matrix(self, R, rs, e):
    """Calculates gaussian distance matrix.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors

    Parameters
    ----------
      R [B, N, M]: tf.Tensor
        Distance matrix.
      rs: tf.Variable
        Gaussian distance matrix mean.
      e: tf.Variable
        Gaussian distance matrix width (e = .5/std**2).

    Returns
    -------
      retval [B, N, M]: tf.Tensor
        Gaussian distance matrix.
    """
    return tf.exp(-e * (R - rs)**2)

  def distance_tensor(self, X, Nbrs, boxsize, B, N, M, d):
    """Calculates distance tensor for batch of molecules.

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features tensor.
    Nbrs: tf.Tensor of shape (B, N, M)
      Neighbor list tensor.
    boxsize: float or None
      Simulation box length [Angstrom].

    Returns
    -------
    D: tf.Tensor of shape (B, N, M, d)
      Coordinates/features distance tensor.
    """
    D = []
    for coords, neighbors in zip(tf.unstack(X), tf.unstack(Nbrs)):
      flat_neighbors = tf.reshape(neighbors, [-1])
      neighbor_coords = tf.gather(coords, flat_neighbors)
      neighbor_coords = tf.reshape(neighbor_coords, [N, M, d])
      D.append(neighbor_coords - tf.expand_dims(coords, 1))
    D = tf.stack(D)
    if boxsize is not None:
      boxsize = tf.reshape(boxsize, [1, 1, 1, d])
      D -= tf.round(D / boxsize) * boxsize
    return D

  def distance_matrix(self, D):
    """Calcuates the distance matrix from the distance tensor

    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

    Parameters
    ----------
    D: tf.Tensor of shape (B, N, M, d)
      Distance tensor.

    Returns
    -------
    R: tf.Tensor of shape (B, N, M)
       Distance matrix.
    """
    R = tf.reduce_sum(tf.multiply(D, D), 3)
    R = tf.sqrt(R)
    return R


class AlphaShareLayer(tf.keras.layers.Layer):
  """
  Part of a sluice network. Adds alpha parameters to control
  sharing between the main and auxillary tasks

  Factory method AlphaShare should be used for construction

  Parameters
  ----------
  in_layers: list of Layers or tensors
    tensors in list must be the same size and list must include two or more tensors

  Returns
  -------
  out_tensor: a tensor with shape [len(in_layers), x, y] where x, y were the original layer dimensions
  Distance matrix.
  """

  def __init__(self, **kwargs):
    super(AlphaShareLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    n_alphas = 2*len(input_shape)
    self.alphas = tf.Variable(
          tf.random_normal([n_alphas, n_alphas]), name='alphas')

  def call(self, inputs):
    # check that there isnt just one or zero inputs
    if len(inputs) <= 1:
      raise ValueError("AlphaShare must have more than one input")
    self.num_outputs = len(inputs)
    # create subspaces
    subspaces = []
    original_cols = int(inputs[0].get_shape()[-1].value)
    subspace_size = int(original_cols / 2)
    for input_tensor in inputs:
      subspaces.append(tf.reshape(input_tensor[:, :subspace_size], [-1]))
      subspaces.append(tf.reshape(input_tensor[:, subspace_size:], [-1]))
    n_alphas = len(subspaces)
    subspaces = tf.reshape(tf.stack(subspaces), [n_alphas, -1])
    subspaces = tf.matmul(self.alphas, subspaces)

    # concatenate subspaces, reshape to size of original input, then stack
    # such that out_tensor has shape (2,?,original_cols)
    count = 0
    out_tensors = []
    tmp_tensor = []
    for row in range(n_alphas):
      tmp_tensor.append(tf.reshape(subspaces[row,], [-1, subspace_size]))
      count += 1
      if (count == 2):
        out_tensors.append(tf.concat(tmp_tensor, 1))
        tmp_tensor = []
        count = 0
    return out_tensors


class SluiceLoss(tf.keras.layers.Layer):
  """
  Calculates the loss in a Sluice Network
  Every input into an AlphaShare should be used in SluiceLoss
  """

  def __init__(self, **kwargs):
    super(SluiceLoss, self).__init__(**kwargs)

  def call(self, inputs):
    temp = []
    subspaces = []
    # creates subspaces the same way it was done in AlphaShare
    for input_tensor in inputs:
      subspace_size = int(input_tensor.get_shape()[-1].value / 2)
      subspaces.append(input_tensor[:, :subspace_size])
      subspaces.append(input_tensor[:, subspace_size:])
      product = tf.matmul(tf.transpose(subspaces[0]), subspaces[1])
      subspaces = []
      # calculate squared Frobenius norm
      temp.append(tf.reduce_sum(tf.pow(product, 2)))
    return tf.reduce_sum(temp)


class BetaShare(tf.keras.layers.Layer):
  """
  Part of a sluice network. Adds beta params to control which layer
  outputs are used for prediction

  Parameters
  ----------
  in_layers: list of Layers or tensors
    tensors in list must be the same size and list must include two or more tensors

  Returns
  -------
  output_layers: list of Layers or tensors with same size as in_layers
    Distance matrix.
  """

  def __init__(self, **kwargs):
    super(BetaShare, self).__init__(**kwargs)

  def build(self, input_shape):
    n_betas = len(input_shape)
    self.betas = tf.Variable(tf.random_normal([1, n_betas]), name='betas')

  def call(self, inputs):
    """
    Size of input layers must all be the same
    """
    subspaces = []
    original_cols = int(inputs[0].get_shape()[-1].value)
    for input_tensor in inputs:
      subspaces.append(tf.reshape(input_tensor, [-1]))
    n_betas = len(inputs)
    subspaces = tf.reshape(tf.stack(subspaces), [n_betas, -1])
    out_tensor = tf.matmul(self.betas, subspaces)
    return tf.reshape(out_tensor, [-1, original_cols])


class ANIFeat(tf.keras.layers.Layer):
  """Performs transform from 3D coordinates to ANI symmetry functions
  """

  def __init__(self,
               max_atoms=23,
               radial_cutoff=4.6,
               angular_cutoff=3.1,
               radial_length=32,
               angular_length=8,
               atom_cases=[1, 6, 7, 8, 16],
               atomic_number_differentiated=True,
               coordinates_in_bohr=True,
               **kwargs):
    """
    Only X can be transformed
    """
    super(ANIFeat, self).__init__(**kwargs)
    self.max_atoms = max_atoms
    self.radial_cutoff = radial_cutoff
    self.angular_cutoff = angular_cutoff
    self.radial_length = radial_length
    self.angular_length = angular_length
    self.atom_cases = atom_cases
    self.atomic_number_differentiated = atomic_number_differentiated
    self.coordinates_in_bohr = coordinates_in_bohr

  def call(self, inputs):
    """
    In layers should be of shape dtype tf.float32, (None, self.max_atoms, 4)

    """
    atom_numbers = tf.cast(inputs[0][:, :, 0], tf.int32)
    flags = tf.sign(atom_numbers)
    flags = tf.cast(
        tf.expand_dims(flags, 1) * tf.expand_dims(flags, 2), tf.float32)
    coordinates = inputs[0][:, :, 1:]
    if self.coordinates_in_bohr:
      coordinates = coordinates * 0.52917721092

    d = self.distance_matrix(coordinates, flags)

    d_radial_cutoff = self.distance_cutoff(d, self.radial_cutoff, flags)
    d_angular_cutoff = self.distance_cutoff(d, self.angular_cutoff, flags)

    radial_sym = self.radial_symmetry(d_radial_cutoff, d, atom_numbers)
    angular_sym = self.angular_symmetry(d_angular_cutoff, d, atom_numbers,
                                        coordinates)
    return tf.concat(
        [
            tf.cast(tf.expand_dims(atom_numbers, 2), tf.float32), radial_sym,
            angular_sym
        ],
        axis=2)

  def distance_matrix(self, coordinates, flags):
    """ Generate distance matrix """
    # (TODO YTZ:) faster, less memory intensive way
    # r = tf.reduce_sum(tf.square(coordinates), 2)
    # r = tf.expand_dims(r, -1)
    # inner = 2*tf.matmul(coordinates, tf.transpose(coordinates, perm=[0,2,1]))
    # # inner = 2*tf.matmul(coordinates, coordinates, transpose_b=True)

    # d = r - inner + tf.transpose(r, perm=[0,2,1])
    # d = tf.nn.relu(d) # fix numerical instabilities about diagonal
    # d = tf.sqrt(d) # does this have negative elements? may be unstable for diagonals

    max_atoms = self.max_atoms
    tensor1 = tf.stack([coordinates] * max_atoms, axis=1)
    tensor2 = tf.stack([coordinates] * max_atoms, axis=2)

    # Calculate pairwise distance
    d = tf.sqrt(
        tf.reduce_sum(tf.squared_difference(tensor1, tensor2), axis=3) + 1e-7)

    d = d * flags
    return d

  def distance_cutoff(self, d, cutoff, flags):
    """ Generate distance matrix with trainable cutoff """
    # Cutoff with threshold Rc
    d_flag = flags * tf.sign(cutoff - d)
    d_flag = tf.nn.relu(d_flag)
    d_flag = d_flag * tf.expand_dims((1 - tf.eye(self.max_atoms)), 0)
    d = 0.5 * (tf.cos(np.pi * d / cutoff) + 1)
    return d * d_flag
    # return d

  def radial_symmetry(self, d_cutoff, d, atom_numbers):
    """ Radial Symmetry Function """
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.radial_cutoff, self.radial_length)
    ita = np.ones_like(Rs) * 3 / (Rs[1] - Rs[0])**2
    Rs = tf.cast(np.reshape(Rs, (1, 1, 1, -1)), tf.float32)
    ita = tf.cast(np.reshape(ita, (1, 1, 1, -1)), tf.float32)
    length = ita.get_shape().as_list()[-1]

    d_cutoff = tf.stack([d_cutoff] * length, axis=3)
    d = tf.stack([d] * length, axis=3)

    out = tf.exp(-ita * tf.square(d - Rs)) * d_cutoff
    if self.atomic_number_differentiated:
      out_tensors = []
      for atom_type in self.atom_cases:
        selected_atoms = tf.expand_dims(
            tf.expand_dims(atom_numbers_embedded[:, :, atom_type], axis=1),
            axis=3)
        out_tensors.append(tf.reduce_sum(out * selected_atoms, axis=2))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out, axis=2)

  def angular_symmetry(self, d_cutoff, d, atom_numbers, coordinates):
    """ Angular Symmetry Function """

    max_atoms = self.max_atoms
    embedding = tf.eye(np.max(self.atom_cases) + 1)
    atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

    Rs = np.linspace(0., self.angular_cutoff, self.angular_length)
    ita = 3 / (Rs[1] - Rs[0])**2
    thetas = np.linspace(0., np.pi, self.angular_length)
    zeta = float(self.angular_length**2)

    ita, zeta, Rs, thetas = np.meshgrid(ita, zeta, Rs, thetas)
    zeta = tf.cast(np.reshape(zeta, (1, 1, 1, 1, -1)), tf.float32)
    ita = tf.cast(np.reshape(ita, (1, 1, 1, 1, -1)), tf.float32)
    Rs = tf.cast(np.reshape(Rs, (1, 1, 1, 1, -1)), tf.float32)
    thetas = tf.cast(np.reshape(thetas, (1, 1, 1, 1, -1)), tf.float32)
    length = zeta.get_shape().as_list()[-1]

    # tf.stack issues again...
    vector_distances = tf.stack([coordinates] * max_atoms, 1) - tf.stack(
        [coordinates] * max_atoms, 2)
    R_ij = tf.stack([d] * max_atoms, axis=3)
    R_ik = tf.stack([d] * max_atoms, axis=2)
    f_R_ij = tf.stack([d_cutoff] * max_atoms, axis=3)
    f_R_ik = tf.stack([d_cutoff] * max_atoms, axis=2)

    # Define angle theta = arccos(R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance))
    vector_mul = tf.reduce_sum(tf.stack([vector_distances] * max_atoms, axis=3) * \
                               tf.stack([vector_distances] * max_atoms, axis=2), axis=4)
    vector_mul = vector_mul * tf.sign(f_R_ij) * tf.sign(f_R_ik)
    theta = tf.acos(tf.math.divide(vector_mul, R_ij * R_ik + 1e-5))

    R_ij = tf.stack([R_ij] * length, axis=4)
    R_ik = tf.stack([R_ik] * length, axis=4)
    f_R_ij = tf.stack([f_R_ij] * length, axis=4)
    f_R_ik = tf.stack([f_R_ik] * length, axis=4)
    theta = tf.stack([theta] * length, axis=4)

    out_tensor = tf.pow((1. + tf.cos(theta - thetas)) / 2., zeta) * \
                 tf.exp(-ita * tf.square((R_ij + R_ik) / 2. - Rs)) * f_R_ij * f_R_ik * 2

    if self.atomic_number_differentiated:
      out_tensors = []
      for id_j, atom_type_j in enumerate(self.atom_cases):
        for atom_type_k in self.atom_cases[id_j:]:
          selected_atoms = tf.stack([atom_numbers_embedded[:, :, atom_type_j]] * max_atoms, axis=2) * \
                           tf.stack([atom_numbers_embedded[:, :, atom_type_k]] * max_atoms, axis=1)
          selected_atoms = tf.expand_dims(
              tf.expand_dims(selected_atoms, axis=1), axis=4)
          out_tensors.append(
              tf.reduce_sum(out_tensor * selected_atoms, axis=(2, 3)))
      return tf.concat(out_tensors, axis=2)
    else:
      return tf.reduce_sum(out_tensor, axis=(2, 3))

  def get_num_feats(self):
    n_feat = self.outputs.get_shape().as_list()[-1]
    return n_feat


class GraphEmbedPoolLayer(tf.keras.layers.Layer):
  """
  GraphCNNPool Layer from Robust Spatial Filtering with Graph Convolutional Neural Networks
  https://arxiv.org/abs/1703.00792

  This is a learnable pool operation
  It constructs a new adjacency matrix for a graph of specified number of nodes.

  This differs from our other pool opertions which set vertices to a function value
  without altering the adjacency matrix.

  $V_{emb} = SpatialGraphCNN({V_{in}})$\\
  $V_{out} = \sigma(V_{emb})^{T} * V_{in}$
  $A_{out} = V_{emb}^{T} * A_{in} * V_{emb}$

  """

  def __init__(self, num_vertices, **kwargs):
    self.num_vertices = num_vertices
    super(GraphEmbedPoolLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    no_features = int(input_shape[0][-1])
    self.W = tf.Variable(
        tf.truncated_normal(
            [no_features, self.num_vertices], stddev=1.0 / np.sqrt(no_features)),
        name='weights',
        dtype=tf.float32)
    self.b = tf.Variable(tf.constant(0.1), name='bias', dtype=tf.float32)

  def call(self, inputs):
    """
    Parameters
    ----------
    num_filters: int
      Number of filters to have in the output

    in_layers: list of Layers or tensors
      [V, A, mask]
      V are the vertex features must be of shape (batch, vertex, channel)

      A are the adjacency matrixes for each graph
        Shape (batch, from_vertex, adj_matrix, to_vertex)

      mask is optional, to be used when not every graph has the
      same number of vertices

    Returns: tf.tensor
    Returns a tf.tensor with a graph convolution applied
    The shape will be (batch, vertex, self.num_filters)
    """
    if len(inputs) == 3:
      V, A, mask = inputs
    else:
      V, A = inputs
      mask = None
    factors = self.embedding_factors(V)

    if mask is not None:
      factors = tf.multiply(factors, mask)
    factors = self.softmax_factors(factors)

    result = tf.matmul(factors, V, transpose_a=True)

    result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
    result_A = tf.matmul(result_A, factors)
    result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
    result_A = tf.matmul(factors, result_A, transpose_a=True)
    result_A = tf.reshape(result_A, (tf.shape(A)[0], self.num_vertices,
                                     A.get_shape()[2].value, self.num_vertices))
    return result, result_A

  def embedding_factors(self, V):
    no_features = V.get_shape()[-1].value
    V_reshape = tf.reshape(V, (-1, no_features))
    s = tf.slice(tf.shape(V), [0], [len(V.get_shape()) - 1])
    s = tf.concat([s, tf.stack([self.num_vertices])], 0)
    result = tf.reshape(tf.matmul(V_reshape, self.W) + self.b, s)
    return result

  def softmax_factors(self, V, axis=1):
    max_value = tf.reduce_max(V, axis=axis, keepdims=True)
    exp = tf.exp(tf.subtract(V, max_value))
    prob = tf.math.divide(exp, tf.reduce_sum(exp, axis=axis, keepdims=True))
    return prob


class GraphCNN(tf.keras.layers.Layer):
  """
  GraphCNN Layer from Robust Spatial Filtering with Graph Convolutional Neural Networks
  https://arxiv.org/abs/1703.00792

  Spatial-domain convolutions can be defined as
  H = h_0I + h_1A + h_2A^2 + ... + hkAk, H ∈ R**(N×N)

  We approximate it by
  H ≈ h_0I + h_1A

  We can define a convolution as applying multiple these linear filters
  over edges of different types (think up, down, left, right, diagonal in images)
  Where each edge type has its own adjacency matrix
  H ≈ h_0I + h_1A_1 + h_2A_2 + . . . h_(L−1)A_(L−1)

  V_out = \sum_{c=1}^{C} H^{c} V^{c} + b
  """

  def __init__(self, num_filters, **kwargs):
    """
    Parameters
    ----------
    num_filters: int
      Number of filters to have in the output

    in_layers: list of Layers or tensors
      [V, A, mask]
      V are the vertex features must be of shape (batch, vertex, channel)

      A are the adjacency matrixes for each graph
        Shape (batch, from_vertex, adj_matrix, to_vertex)

      mask is optional, to be used when not every graph has the
      same number of vertices

    Returns: tf.tensor
    Returns a tf.tensor with a graph convolution applied
    The shape will be (batch, vertex, self.num_filters)
    """
    super(GraphCNN, self).__init__(**kwargs)
    self.num_filters = num_filters

  def build(self, input_shape):
    no_features = int(input_shape[0][2])
    no_A = int(input_shape[1][2])
    self.W = tf.Variable(
        tf.truncated_normal(
            [no_features * no_A, self.num_filters],
            stddev=np.sqrt(1.0 / (no_features * (no_A + 1) * 1.0))),
        name='weights',
        dtype=tf.float32)
    self.W_I = tf.Variable(
        tf.truncated_normal(
            [no_features, self.num_filters],
            stddev=np.sqrt(1.0 / (no_features * (no_A + 1) * 1.0))),
        name='weights_I',
        dtype=tf.float32)
    self.b = tf.Variable(tf.constant(0.1), name='bias', dtype=tf.float32)

  def call(self, inputs):
    if len(inputs) == 3:
      V, A, mask = inputs
    else:
      V, A = inputs
    no_A = A.get_shape()[2].value
    no_features = V.get_shape()[2].value
    n = self.graphConvolution(V, A)
    A_shape = tf.shape(A)
    n = tf.reshape(n, [-1, A_shape[1], no_A * no_features])
    return self.batch_mat_mult(n, self.W) + self.batch_mat_mult(V, self.W_I) + self.b

  def graphConvolution(self, V, A):
    no_A = A.get_shape()[2].value
    no_features = V.get_shape()[2].value
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1] * no_A, A_shape[1]]))
    n = tf.matmul(A_reshape, V)
    return tf.reshape(n, [-1, A_shape[1], no_A, no_features])

  def batch_mat_mult(self, A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    # So the Tensor has known dimensions
    if B.get_shape()[1] == None:
      axis_2 = -1
    else:
      axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    return tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
