import random
import string

import tensorflow as tf

from deepchem.nn import model_ops, initializations


class Layer(object):

  def __init__(self, in_layers=None, **kwargs):
    if "name" not in kwargs:
      self.name = "%s%s" % (self.__class__.__name__, self._random_name())
    else:
      self.name = kwargs['name']
    if "tensorboard" not in kwargs:
      self.tensorboard = False
    else:
      self.tensorboard = kwargs['tensorboard']
    if in_layers is None:
      in_layers = list()
    self.in_layers = in_layers

  def _random_name(self):
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(4))

  def none_tensors(self):
    out_tensor = self.out_tensor
    self.out_tensor = None
    return out_tensor

  def set_tensors(self, tensor):
    self.out_tensor = tensor

  def _create_tensor(self):
    raise ValueError("Subclasses must implement for themselves")

  def __key(self):
    return self.name

  def __eq__(x, y):
    if x is None or y is None:
      return False
    if type(x) != type(y):
      return False
    return x.__key() == y.__key()

  def __hash__(self):
    return hash(self.__key())

  def __call__(self, *in_layers):
    if len(in_layers) > 0:
      layers = []
      for in_layer in in_layers:
        if isinstance(in_layer, Layer):
          layers.append(layer)
        elif isinstance(in_layer, tf.Tensor):
          layers.append(TensorWrapper(in_layer))
        else:
          raise ValueError("Layer must be invoked on layers or tensors")
      self.in_layers = layers
    self._create_tensor()
  

class TensorWrapper(Layer):
  """Used to wrap a tensorflow tensor."""
  def __init__(self, out_tensor):
    self.out_tensor = out_tensor


class Conv1DLayer(Layer):

  def __init__(self, width, out_channels, **kwargs):
    self.width = width
    self.out_channels = out_channels
    self.out_tensor = None
    super(Conv1DLayer, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Only One Parent to conv1D over")
    parent = self.in_layers[0]
    if len(parent.out_tensor.get_shape()) != 3:
      raise ValueError("Parent tensor must be (batch, width, channel)")
    parent_shape = parent.out_tensor.get_shape()
    parent_channel_size = parent_shape[2].value
    f = tf.Variable(
        tf.random_normal([self.width, parent_channel_size, self.out_channels]))
    b = tf.Variable(tf.random_normal([self.out_channels]))
    t = tf.nn.conv1d(parent.out_tensor, f, stride=1, padding="SAME")
    t = tf.nn.bias_add(t, b)
    self.out_tensor = tf.nn.relu(t)
    return self.out_tensor


class Dense(Layer):

  def __init__(self, out_channels, activation_fn=None, **kwargs):
    self.out_channels = out_channels
    self.out_tensor = None
    self.activation_fn = activation_fn
    super(Dense, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Only One Parent to Dense over %s" % self.in_layers)
    parent = self.in_layers[0]
    if len(parent.out_tensor.get_shape()) != 2:
      raise ValueError("Parent tensor must be (batch, width)")
    in_channels = parent.out_tensor.get_shape()[-1].value
    # w = initializations.glorot_uniform([in_channels, self.out_channels])
    # w = model_ops.zeros(shape=[in_channels, self.out_channels])
    # b = tf.Variable([0.0, 0.0])
    # self.out_tensor = tf.matmul(parent.out_tensor, w) + b
    self.out_tensor = tf.contrib.layers.fully_connected(
        parent.out_tensor,
        num_outputs=self.out_channels,
        activation_fn=self.activation_fn,
        scope=self.name,
        trainable=True)
    return self.out_tensor


class Flatten(Layer):

  def __init__(self, **kwargs):
    super(Flatten, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Only One Parent to conv1D over")
    parent = self.in_layers[0]
    parent_shape = parent.out_tensor.get_shape()
    vector_size = 1
    for i in range(1, len(parent_shape)):
      vector_size *= parent_shape[i].value
    parent_tensor = parent.out_tensor
    self.out_tensor = tf.reshape(parent_tensor, shape=(-1, vector_size))
    return self.out_tensor


class Reshape(Layer):

  def __init__(self, shape, **kwargs):
    self.shape = shape
    super(Reshape, self).__init__(**kwargs)

  def _create_tensor(self):
    parent_tensor = self.in_layers[0].out_tensor
    self.out_tensor = tf.reshape(parent_tensor, self.shape)


class CombineMeanStd(Layer):

  def __init__(self, **kwargs):
    super(CombineMeanStd, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 2:
      raise ValueError("Must have two self.in_layers")
    mean_parent, std_parent = self.in_layers[0], self.in_layers[1]
    mean_parent_tensor, std_parent_tensor = mean_parent.out_tensor, std_parent.out_tensor
    sample_noise = tf.random_normal(
        mean_parent_tensor.get_shape(), 0, 1, dtype=tf.float32)
    self.out_tensor = mean_parent_tensor + (std_parent_tensor * sample_noise)


class Repeat(Layer):

  def __init__(self, n_times, **kwargs):
    self.n_times = n_times
    super(Repeat, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = self.in_layers[0].out_tensor
    t = tf.expand_dims(parent_tensor, 1)
    pattern = tf.stack([1, self.n_times, 1])
    self.out_tensor = tf.tile(t, pattern)


class GRU(Layer):

  def __init__(self, n_hidden, out_channels, batch_size, **kwargs):
    self.n_hidden = n_hidden
    self.out_channels = out_channels
    self.batch_size = batch_size
    super(GRU, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = self.in_layers[0].out_tensor
    gru_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
    initial_gru_state = gru_cell.zero_state(self.batch_size, tf.float32)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        gru_cell,
        parent_tensor,
        initial_state=initial_gru_state,
        scope=self.name)
    projection = lambda x: tf.contrib.layers.linear(x, num_outputs=self.out_channels, activation_fn=tf.nn.sigmoid)
    self.out_tensor = tf.map_fn(projection, rnn_outputs)


class TimeSeriesDense(Layer):

  def __init__(self, out_channels, **kwargs):
    self.out_channels = out_channels
    super(TimeSeriesDense, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = self.in_layers[0].out_tensor
    dense_fn = lambda x: tf.contrib.layers.fully_connected(x, num_outputs=self.out_channels,
                                                           activation_fn=tf.nn.sigmoid)
    self.out_tensor = tf.map_fn(dense_fn, parent_tensor)


class Input(Layer):

  def __init__(self, shape, dtype=tf.float32, **kwargs):
    self.shape = shape
    self.dtype = dtype
    super(Input, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) > 0:
      queue = self.in_layers[0]
      placeholder = queue.out_tensors[self.get_pre_q_name()]
      self.out_tensor = tf.placeholder_with_default(placeholder, self.shape)
      return self.out_tensor
    self.out_tensor = tf.placeholder(dtype=self.dtype, shape=self.shape)
    return self.out_tensor

  def create_pre_q(self, batch_size):
    q_shape = (batch_size,) + self.shape[1:]
    return Input(shape=q_shape, name="%s_pre_q" % self.name, dtype=self.dtype)

  def get_pre_q_name(self):
    return "%s_pre_q" % self.name


class Feature(Input):

  def __init__(self, **kwargs):
    super(Feature, self).__init__(**kwargs)


class Label(Input):

  def __init__(self, **kwargs):
    super(Label, self).__init__(**kwargs)


class Weights(Input):

  def __init__(self, **kwargs):
    super(Weights, self).__init__(**kwargs)


class L2LossLayer(Layer):

  def __init__(self, **kwargs):
    super(L2LossLayer, self).__init__(**kwargs)

  def _create_tensor(self):
    guess, label = self.in_layers[0], self.in_layers[1]
    self.out_tensor = tf.reduce_mean(
        tf.square(guess.out_tensor - label.out_tensor))
    return self.out_tensor


class SoftMax(Layer):

  def __init__(self, **kwargs):
    super(SoftMax, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Must only Softmax single parent")
    parent = self.in_layers[0]
    self.out_tensor = tf.contrib.layers.softmax(parent.out_tensor)
    return self.out_tensor


class Concat(Layer):

  def __init__(self, axis=1, **kwargs):
    self.axis = axis
    super(Concat, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) == 1:
      self.out_tensor = self.in_layers[0].out_tensor
      return self.out_tensor
    out_tensors = [x.out_tensor for x in self.in_layers]

    self.out_tensor = tf.concat(out_tensors, axis=self.axis)
    return self.out_tensor


class InteratomicL2Distances(Layer):
  """Compute (squared) L2 Distances between atoms given neighbors."""

  def _create_tensor(self):
    if len(self.in_layers) != 2:
      raise ValueError("InteratomicDistances requires coords,nbr_list")
    coords, nbr_list = (self.in_layers[0].out_tensor,
                        self.in_layers[1].out_tensor)
    N_atoms, ndim = coords.get_shape()
    _, M = nbr_list.get_shape()
    # Shape (N_atoms, M, ndim)
    nbr_coords = tf.gather(coords, nbr_list)
    # Shape (N_atoms, M, ndim)
    tiled_atom_coords = tf.tile(
        tf.reshape(atom_coords, (N_atoms, 1, ndim)), (1, M, 1))
    # Shape (N_atoms, M)
    dists = tf.reduce_sum((tiled_atom_coords - nbr_coords)**2, axis=2)



class SoftMaxCrossEntropy(Layer):

  def __init__(self, **kwargs):
    super(SoftMaxCrossEntropy, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) != 2:
      raise ValueError()
    labels, logits = self.in_layers[0].out_tensor, self.in_layers[1].out_tensor
    self.out_tensor = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    self.out_tensor = tf.reshape(self.out_tensor, [-1, 1])
    return self.out_tensor


class ReduceMean(Layer):

  def __init__(self, axis=None, **kwargs):
    self.axis=axis
    super(ReduceMean, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) > 1:
      out_tensors = [x.out_tensor for x in self.in_layers]
      self.out_tensor = tf.stack(out_tensors)
    else:
      self.out_tensor = self.in_layers[0].out_tensor

    self.out_tensor = tf.reduce_mean(self.out_tensor)
    return self.out_tensor

class ToFloat(Layer):
  def _create_tensor(self):
    if len(self.in_layers) > 1:
      raise ValueError("Only one layer supported.")
    self.out_tensor = tf.to_float(self.in_layers[0].out_tensor)
    return self.out_tensor

class ReduceSum(Layer):

  def __init__(self, axis=None, **kwargs):
    self.axis=axis
    super(ReduceSum, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) > 1:
      out_tensors = [x.out_tensor for x in self.in_layers]
      self.out_tensor = tf.stack(out_tensors)
    else:
      self.out_tensor = self.in_layers[0].out_tensor

    self.out_tensor = tf.reduce_sum(self.out_tensor, axis=self.axis)
    return self.out_tensor


class ReduceSquareDifference(Layer):

  def __init__(self, axis=None, **kwargs):
    self.axis=axis
    super(ReduceSquareDifference, self).__init__(**kwargs)

  def _create_tensor(self):
    a = self.in_layers[0].out_tensor
    b = self.in_layers[1].out_tensor
    self.out_tensor = tf.reduce_mean(tf.squared_difference(a, b),
                                     axis=self.axis)
    return self.out_tensor


class Conv2d(Layer):

  def __init__(self, num_outputs, kernel_size=5, **kwargs):
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    super(Conv2d, self).__init__(**kwargs)

  def _create_tensor(self):
    parent_tensor = self.in_layers[0].out_tensor
    out_tensor = tf.contrib.layers.conv2d(
        parent_tensor,
        num_outputs=self.num_outputs,
        kernel_size=self.kernel_size,
        padding="SAME",
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.contrib.layers.batch_norm)
    self.out_tensor = out_tensor


class MaxPool(Layer):

  def __init__(self,
               ksize=[1, 2, 2, 1],
               strides=[1, 2, 2, 1],
               padding="SAME",
               **kwargs):
    self.ksize = ksize
    self.strides = strides
    self.padding = padding
    super(MaxPool, self).__init__(**kwargs)

  def _create_tensor(self):
    in_tensor = self.in_layers[0].out_tensor
    self.out_tensor = tf.nn.max_pool(
        in_tensor, ksize=self.ksize, strides=self.strides, padding=self.padding)
    return self.out_tensor


class InputFifoQueue(Layer):
  """
  This Queue Is used to allow asynchronous batching of inputs
  During the fitting process
  """

  def __init__(self, shapes, names, dtypes=None, capacity=5, **kwargs):
    self.shapes = shapes
    self.names = names
    self.capacity = capacity
    self.dtypes = dtypes
    super(InputFifoQueue, self).__init__(**kwargs)

  def _create_tensor(self):
    if self.dtypes is None:
      self.dtypes = [tf.float32] * len(self.shapes)
    self.queue = tf.FIFOQueue(
        self.capacity, self.dtypes, shapes=self.shapes, names=self.names)
    feed_dict = {x.name: x.out_tensor for x in self.in_layers}
    self.out_tensor = self.queue.enqueue(feed_dict)
    self.close_op = self.queue.close()
    self.out_tensors = self.queue.dequeue()

  def none_tensors(self):
    queue, out_tensors, out_tensor, close_op = self.queue, self.out_tensor, self.out_tensor, self.close_op
    self.queue, self.out_tensor, self.out_tensors, self.close_op = None, None, None, None
    return queue, out_tensors, out_tensor, close_op

  def set_tensors(self, tensors):
    self.queue, self.out_tensor, self.out_tensors, self.close_op = tensors

  def close(self):
    self.queue.close()


class GraphConvLayer(Layer):

  def __init__(self,
               out_channel,
               min_deg=0,
               max_deg=10,
               activation_fn=None,
               **kwargs):
    self.out_channel = out_channel
    self.min_degree = min_deg
    self.max_degree = max_deg
    self.num_deg = 2 * max_deg + (1 - min_deg)
    self.activation_fn = activation_fn
    super(GraphConvLayer, self).__init__(**kwargs)

  def _create_tensor(self):
    #   self.in_layers = [atom_features, deg_slice, membership, deg_adj_list placeholders...]
    in_channels = self.in_layers[0].out_tensor.get_shape()[-1].value

    # Generate the nb_affine weights and biases
    self.W_list = [
        initializations.glorot_uniform([in_channels, self.out_channel])
        for k in range(self.num_deg)
    ]
    self.b_list = [
        model_ops.zeros(shape=[
            self.out_channel,
        ]) for k in range(self.num_deg)
    ]

    # Extract atom_features
    atom_features = self.in_layers[0].out_tensor

    # Extract graph topology
    deg_slice = self.in_layers[1].out_tensor
    deg_adj_lists = [x.out_tensor for x in self.in_layers[3:]]

    # Perform the mol conv
    # atom_features = graph_conv(atom_features, deg_adj_lists, deg_slice,
    #                            self.max_deg, self.min_deg, self.W_list,
    #                            self.b_list)

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

    self.out_tensor = atom_features
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

  def none_tensors(self):
    out_tensor, W_list, b_list = self.out_tensor, self.W_list, self.b_list
    self.out_tensor, self.W_list, self.b_list = None, None, None
    return out_tensor, W_list, b_list

  def set_tensors(self, tensors):
    self.out_tensor, self.W_list, self.b_list = tensors


class GraphPoolLayer(Layer):

  def __init__(self, min_degree=0, max_degree=10, **kwargs):
    self.min_degree = min_degree
    self.max_degree = max_degree
    super(GraphPoolLayer, self).__init__(**kwargs)

  def _create_tensor(self):
    atom_features = self.in_layers[0].out_tensor
    deg_slice = self.in_layers[1].out_tensor
    deg_adj_lists = [x.out_tensor for x in self.in_layers[3:]]

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

    self.out_tensor = tf.concat(axis=0, values=deg_maxed)
    return self.out_tensor


class GraphGather(Layer):

  def __init__(self, batch_size, activation_fn=None, **kwargs):
    self.batch_size = batch_size
    self.activation_fn = activation_fn
    super(GraphGather, self).__init__(**kwargs)

  def _create_tensor(self):
    # x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]
    atom_features = self.in_layers[0].out_tensor

    # Extract graph topology
    membership = self.in_layers[2].out_tensor

    # Perform the mol gather

    assert (self.batch_size > 1, "graph_gather requires batches larger than 1")

    # Obtain the partitions for each of the molecules
    activated_par = tf.dynamic_partition(atom_features, membership,
                                         self.batch_size)

    # Sum over atoms for each molecule
    sparse_reps = [
        tf.reduce_mean(activated, 0, keep_dims=True)
        for activated in activated_par
    ]
    max_reps = [
        tf.reduce_max(activated, 0, keep_dims=True)
        for activated in activated_par
    ]

    # Get the final sparse representations
    sparse_reps = tf.concat(axis=0, values=sparse_reps)
    max_reps = tf.concat(axis=0, values=max_reps)
    mol_features = tf.concat(axis=1, values=[sparse_reps, max_reps])

    if self.activation_fn is not None:
      mol_features = self.activation_fn(mol_features)
    self.out_tensor = mol_features
    return mol_features


class BatchNormLayer(Layer):

  def _create_tensor(self):
    parent_tensor = self.in_layers[0].out_tensor
    self.out_tensor = tf.layers.batch_normalization(parent_tensor)
    return self.out_tensor


class WeightedError(Layer):

  def _create_tensor(self):
    entropy, weights = self.in_layers[0], self.in_layers[1]
    self.out_tensor = tf.reduce_sum(entropy.out_tensor * weights.out_tensor)
    return self.out_tensor

class Cutoff(Layer):
  """Truncates interactions that are too far away."""
  def __init__(dist, **kwargs):
    self.d = dist
    super(Cutoff, self).__init__(**kwargs)
  
  
  def _create_tensor(self):
    d = self.d
    x = self.in_layers[0].out_tensor
    self.out_tensor = tf.where(d < 8, x, tf.zeros_like(x))
    return self.out_tensor

class VinaRepulsion(Layer):
  """Computes Autodock Vina's repulsion interaction term."""
  
  def _create_tensor(self):
    d = self.in_layers[0].out_tensor
    self.out_tensor = tf.where(d < 0, d**2, tf.zeros_like(d))
    return self.out_tensor

def VinaHydrophobic(Layer):
  """Computes Autodock Vina's hydrophobic interaction term."""

  def _create_tensor(self):
    d = self.in_layers[0].out_tensor
    self.out_tensor = tf.where(d < 0.5,
                               tf.ones_like(d),
                               tf.where(d < 1.5, 1.5 - d, tf.zeros_like(d)))
    return self.out_tensor

class VinaHydrogenBond(Layer):
  """Computes Autodock Vina's hydrogen bond interaction term."""

  def _create_tensor(self):
    d = self.in_layers[0].out_tensor
    self.out_tensor = tf.where(d < -0.7,
                               tf.ones_like(d),
                               tf.where(d < 0,
                                        (1.0 / 0.7) * (0 - d),
                                        tf.zeros_like(d)))
    return self.out_tensor

class VinaGaussianFirst(Layer):
  """Computes Autodock Vina's first Gaussian interaction term."""
  
  def _create_tensor(self):
    d = self.in_layers[0].out_tensor
    self.out_tensor = tf.exp(-(d / 0.5)**2)
    return self.out_tensor

class VinaGaussianSecond(Layer):
  """Computes Autodock Vina's second Gaussian interaction term."""

  def _create_tensor(self):
    d = self.in_layers[0].out_tensor
    self.out_tensor = tf.exp(-((d - 3) / 2)**2)
    return self.out_tensor


class WeightedLinearCombo(Layer):
  """Computes a weighted linear combination of input layers.""" 

  def __init__(self, std=.3, **kwargs):
    self.std = std
    super(WeightedLinearCombo, self).__init__(**kwargs)

  def _create_tensor(self):
    weights = []
    out_tensor = None
    for in_layer in self.in_layers:
      w = tf.Variable(tf.random_normal([1,], stddev=self.std))
      if out_tensor is None:
        out_tensor = w * in_layer.out_tensor
      else:
        out_tensor += w * in_layer.out_tensor
    self.out_tensor = out_tensor
    return self.out_tensor

    
class NeighborList(Layer):
  """Computes a neighbor-list on the GPU.

  Neighbor-lists (also called Verlet Lists) are a tool for grouping atoms which
  are close to each other spatially
  """

  def __init__(self, max_num_atoms, max_num_nbrs, ndim, n_cells, k, nbr_cutoff, **kwargs):
    """
    Parameters
    ----------
    max_num_atoms: int
      Maximum number of atoms this layer will neighbor-list.
    max_num_nbrs: int
      Maximum number of spatial neighbors possible for atom.
    ndim: int
      Dimensionality of space atoms live in. (Typically 3D, but sometimes will
      want to use higher dimensional descriptors for atoms).
    n_cells: int
      Number of grid cells in the simulation box.
    k: int
      Number of nearest neighbors to pull in using tf.nn.top_k.
      TODO(rbharath): Are both k and max_num_nbrs needed?
    nbr_cutoff: float
      Length in Angstroms (?) at which atom boxes are gridded.
    """
    self.N = max_num_atoms
    self.M = max_num_nbrs
    self.ndim = ndim
    self.n_cells = n_cells
    self.k = k
    self.nbr_cutoff = nbr_cutoff
    super(NeighborList, self).__init__(**kwargs)

  def _create_tensor(self):
    """Creates tensors associated with neighbor-listing."""
    if len(self.in_layers) != 1:
      raise ValueError("Only One Parent to NeighborList over %s" % self.in_layers)
    parent = self.in_layers[0]
    if len(parent.out_tensor.get_shape()) != 2:
      # TODO(rbharath): Support batching
      raise ValueError("Parent tensor must be (num_atoms, ndum)")
    coords = parent.out_tensor
    nbr_list = self._compute_nbr_list(coords)
    self.out_tensor = nbr_list
    return nbr_list
  

  def _compute_nbr_list(self, coords):
    """Computes a neighbor list from atom coordinates.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N, ndim)

    Returns
    -------
    nbr_list: tf.Tensor
      Shape (N, M) of atom indices
    """
    N, M, n_cells, ndim, k = self.N, self.M, self.n_cells, self.ndim, self.k
    nbr_cutoff = self.nbr_cutoff
    start = tf.to_int32(tf.reduce_min(coords))
    stop = tf.to_int32(tf.reduce_max(coords))
    cells = self._get_cells(start, stop)
    # Associate each atom with cell it belongs to. O(N*n_cells)
    # Shape (n_cells, k)
    atoms_in_cells, _ = self._put_atoms_in_cells(coords, cells)
    # Shape (N, 1)
    cells_for_atoms = self._get_cells_for_atoms(coords, cells)

    # Associate each cell with its neighbor cells. Assumes periodic boundary   
    # conditions, so does wrapround. O(constant)    
    # Shape (n_cells, 26)
    neighbor_cells = self._compute_neighbor_cells(cells)

    # Shape (N, 26)
    neighbor_cells = tf.squeeze(tf.gather(neighbor_cells, cells_for_atoms))

    # coords of shape (N, ndim)
    # Shape (N, 26, k, ndim)
    tiled_coords = tf.tile(tf.reshape(coords, (N, 1, 1, ndim)), (1, 26, k, 1))

    # Shape (N, 26, k)
    nbr_inds = tf.gather(atoms_in_cells, neighbor_cells)

    # Shape (N, 26, k)
    atoms_in_nbr_cells = tf.gather(atoms_in_cells, neighbor_cells)

    # Shape (N, 26, k, ndim)
    nbr_coords = tf.gather(coords, atoms_in_nbr_cells)

    # For smaller systems especially, the periodic boundary conditions can
    # result in neighboring cells being seen multiple times. Maybe use tf.unique to
    # make sure duplicate neighbors are ignored?

    # TODO(rbharath): How does distance need to be modified here to   
    # account for periodic boundary conditions?   
    # Shape (N, 26, k)
    dists = tf.reduce_sum((tiled_coords - nbr_coords)**2, axis=3)

    # Shape (N, 26*k)
    dists = tf.reshape(dists, [N, -1])

    # TODO(rbharath): This will cause an issue with duplicates!
    # Shape (N, M)
    closest_nbr_locs = tf.nn.top_k(dists, k=M)[1]

    # N elts of size (M,) each
    split_closest_nbr_locs = [
        tf.squeeze(locs) for locs in tf.split(closest_nbr_locs, N)
    ]

    # Shape (N, 26*k)
    nbr_inds = tf.reshape(nbr_inds, [N, -1])

    # N elts of size (26*k,) each
    split_nbr_inds = [tf.squeeze(split) for split in tf.split(nbr_inds, N)]

    # N elts of size (M,) each 
    neighbor_list = [
        tf.gather(nbr_inds, closest_nbr_locs)
        for (nbr_inds, closest_nbr_locs
            ) in zip(split_nbr_inds, split_closest_nbr_locs)
    ]

    # Shape (N, M)
    neighbor_list = tf.stack(neighbor_list)

    return neighbor_list

  def _put_atoms_in_cells(self, coords, cells):
    """Place each atom into cells. O(N) runtime.    
    
    Let N be the number of atoms.
        
    Parameters    
    ----------    
    coords: tf.Tensor 
      (N, 3) shape.
    cells: tf.Tensor
      (n_cells, ndim) shape.
    N: int
      Number atoms
    ndim: int
      Dimensionality of input space
    k: int
      Number of nearest neighbors.

    Returns
    -------
    closest_atoms: tf.Tensor 
      Of shape (n_cells, k, ndim)
    """
    N, n_cells, ndim, k = self.N, self.n_cells, self.ndim, self.k
    # Tile both cells and coords to form arrays of size (n_cells*N, ndim)
    tiled_cells = tf.reshape(tf.tile(cells, (1, N)), (n_cells * N, ndim))
    # TODO(rbharath): Change this for tf 1.0
    # n_cells tensors of shape (N, 1)
    tiled_cells = tf.split(tiled_cells, n_cells)

    # Shape (N*n_cells, 1) after tile
    tiled_coords = tf.tile(coords, (n_cells, 1))
    # List of n_cells tensors of shape (N, 1)
    tiled_coords = tf.split(tiled_coords, n_cells)

    # Lists of length n_cells
    coords_rel = [
        tf.to_float(coords) - tf.to_float(cells)
        for (coords, cells) in zip(tiled_coords, tiled_cells)
    ]
    coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

    # Lists of length n_cells
    # Get indices of k atoms closest to each cell point
    closest_inds = [tf.nn.top_k(norm, k=k)[1] for norm in coords_norm]
    # n_cells tensors of shape (k, ndim)
    closest_atoms = tf.stack([tf.gather(coords, inds) for inds in closest_inds])
    # Tensor of shape (n_cells, k)
    closest_inds = tf.stack(closest_inds)

    return closest_inds, closest_atoms

  def _get_cells_for_atoms(self, coords, cells):
    """Compute the cells each atom belongs to.

    Parameters
    ----------
    coords: tf.Tensor
      Shape (N, ndim)
    cells: tf.Tensor
      (box_size**ndim, ndim) shape.
    Returns
    -------
    cells_for_atoms: tf.Tensor
      Shape (N, 1)
    """
    N, n_cells, ndim = self.N, self.n_cells, self.ndim
    n_cells = int(n_cells)
    # Tile both cells and coords to form arrays of size (n_cells*N, ndim)
    tiled_cells = tf.tile(cells, (N, 1))
    # N tensors of shape (n_cells, 1)
    tiled_cells = tf.split(tiled_cells, N)

    # Shape (N*n_cells, 1) after tile
    tiled_coords = tf.reshape(tf.tile(coords, (1, n_cells)), (n_cells * N, ndim))
    # List of N tensors of shape (n_cells, 1)
    tiled_coords = tf.split(tiled_coords, N)

    # Lists of length N 
    coords_rel = [
        tf.to_float(coords) - tf.to_float(cells)
        for (coords, cells) in zip(tiled_coords, tiled_cells)
    ]
    coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

    # Lists of length n_cells
    # Get indices of k atoms closest to each cell point
    closest_inds = [tf.nn.top_k(-norm, k=1)[1] for norm in coords_norm]

    # TODO(rbharath): tf.stack for tf 1.0
    return tf.stack(closest_inds)

  def _compute_neighbor_cells(self, cells):
    """Compute neighbors of cells in grid.    

    # TODO(rbharath): Do we need to handle periodic boundary conditions
    properly here?
    # TODO(rbharath): This doesn't handle boundaries well. We hard-code
    # looking for 26 neighbors, which isn't right for boundary cells in
    # the cube.
        
    Note n_cells is box_size**ndim. 26 is the number of neighbors of a cube in
    a grid (including diagonals).

    Parameters    
    ----------    
    cells: tf.Tensor
      (n_cells, 26) shape.
    """
    ndim, n_cells = self.ndim, self.n_cells
    n_cells = int(n_cells)
    if ndim != 3:
      raise ValueError("Not defined for dimensions besides 3")
    # Number of neighbors of central cube in 3-space is
    # 3^2 (top-face) + 3^2 (bottom-face) + (3^2-1) (middle-band)
    # TODO(rbharath)
    k = 9 + 9 + 8  # (26 faces on Rubik's cube for example)
    #n_cells = int(cells.get_shape()[0])
    # Tile cells to form arrays of size (n_cells*n_cells, ndim)
    # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
    # Tile (a, a, a, b, b, b, etc.)
    tiled_centers = tf.reshape(
        tf.tile(cells, (1, n_cells)), (n_cells * n_cells, ndim))
    # Tile (a, b, c, a, b, c, ...)
    tiled_cells = tf.tile(cells, (n_cells, 1))

    # Lists of n_cells tensors of shape (N, 1)
    tiled_centers = tf.split(tiled_centers, n_cells)
    tiled_cells = tf.split(tiled_cells, n_cells)

    # Lists of length n_cells
    coords_rel = [
        tf.to_float(cells) - tf.to_float(centers)
        for (cells, centers) in zip(tiled_centers, tiled_cells)
    ]
    coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

    # Lists of length n_cells
    # Get indices of k atoms closest to each cell point
    # n_cells tensors of shape (26,)
    closest_inds = tf.stack([tf.nn.top_k(norm, k=k)[1] for norm in coords_norm])

    return closest_inds

  def _get_cells(self, start, stop):
    """Returns the locations of all grid points in box.

    Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
    Then would return a list of length 20^3 whose entries would be
    [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

    Returns
    -------
    cells: tf.Tensor
      (box_size**ndim, ndim) shape.
    """
    return tf.reshape(
        tf.transpose(
            tf.stack(
                tf.meshgrid(
                    * [tf.range(start, stop, self.nbr_cutoff) for _ in range(self.ndim)]))),
        (-1, self.ndim))

  
