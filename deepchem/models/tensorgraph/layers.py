import random
import string

import tensorflow as tf
import numpy as np

from deepchem.nn import model_ops, initializations


class Layer(object):
  layer_number_dict = {}

  def __init__(self, in_layers=None, **kwargs):
    if "name" not in kwargs:
      self.name = "%s_%s" % (self.__class__.__name__, self._get_layer_number())
    else:
      self.name = kwargs['name']
    if "tensorboard" not in kwargs:
      self.tensorboard = False
    else:
      self.tensorboard = kwargs['tensorboard']
    if in_layers is None:
      in_layers = list()
    self.in_layers = in_layers
    self.op_type = "gpu"

  def _get_layer_number(self):
    class_name = self.__class__.__name__
    if class_name not in Layer.layer_number_dict:
      Layer.layer_number_dict[class_name] = 0
    Layer.layer_number_dict[class_name] += 1
    return "%s" % Layer.layer_number_dict[class_name]

  def none_tensors(self):
    out_tensor = self.out_tensor
    self.out_tensor = None
    return out_tensor

  def set_tensors(self, tensor):
    self.out_tensor = tensor

  def _create_tensor(self):
    raise NotImplementedError("Subclasses must implement for themselves")

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

  def shared(self, in_layers):
    """
    Share weights with different in tensors and a new out tensor
    Parameters
    ----------
    in_layers: list tensor
    List in tensors for the shared layer

    Returns
    -------
    Layer
    """
    raise ValueError("Each Layer must implement shared for itself")


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
  def __init__(
    self,
    out_channels,
    activation_fn=None,
    biases_initializer=tf.zeros_initializer,
    weights_initializer=tf.contrib.layers.variance_scaling_initializer,
    time_series=False,
    scope_name=None,
    reuse=False,
    **kwargs):
    super(Dense, self).__init__(**kwargs)
    self.out_channels = out_channels
    self.out_tensor = None
    self.activation_fn = activation_fn
    self.biases_initializer = biases_initializer
    self.weights_initializer = weights_initializer
    self.time_series = time_series
    self.reuse = reuse
    if scope_name is None:
      scope_name = self.name
    self.scope_name = scope_name

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Only One Parent to Dense over %s" % self.in_layers)
    parent = self.in_layers[0]
    if not self.time_series:
      self.out_tensor = tf.contrib.layers.fully_connected(
        parent.out_tensor,
        num_outputs=self.out_channels,
        activation_fn=self.activation_fn,
        biases_initializer=self.biases_initializer(),
        weights_initializer=self.weights_initializer(),
        scope=self.scope_name,
        reuse=self.reuse,
        trainable=True)
      return self.out_tensor
    dense_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                           num_outputs=self.out_channels,
                                                           activation_fn=self.activation_fn,
                                                           biases_initializer=self.biases_initializer(),
                                                           weights_initializer=self.weights_initializer(),
                                                           scope=self.scope_name,
                                                           reuse=self.reuse,
                                                           trainable=True)
    self.out_tensor = tf.map_fn(dense_fn, parent.out_tensor, back_prop=True)

  def shared(self, in_layers):
    self.reuse = True
    return Dense(
      self.out_channels,
      self.activation_fn,
      self.biases_initializer,
      self.weights_initializer,
      time_series=self.time_series,
      reuse=self.reuse,
      scope_name=self.scope_name,
      in_layers=in_layers)


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


class Transpose(Layer):
  def __init__(self, out_shape, **kwargs):
    super(Transpose, self).__init__(**kwargs)
    self.out_shape = out_shape

  def _create_tensor(self):
    if len(self.in_layers) != 1:
      raise ValueError("Only One Parent to Transpose over")
    self.out_tensor = tf.transpose(self.in_layers[0].out_tensor, self.out_shape)
    return self.out_tensor


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
    self.op_type = "cpu"

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
  def __init__(self, **kwargs):
    super(Concat, self).__init__(**kwargs)

  def _create_tensor(self):
    if len(self.in_layers) == 1:
      self.out_tensor = self.in_layers[0].out_tensor
      return self.out_tensor
    out_tensors = [x.out_tensor for x in self.in_layers]

    self.out_tensor = tf.concat(out_tensors, 1)
    return self.out_tensor


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
  def _create_tensor(self):
    if len(self.in_layers) > 1:
      out_tensors = [x.out_tensor for x in self.in_layers]
      self.out_tensor = tf.stack(out_tensors)
    else:
      self.out_tensor = self.in_layers[0].out_tensor

    self.out_tensor = tf.reduce_mean(self.out_tensor)
    return self.out_tensor


class ReduceSquareDifference(Layer):
  def __init__(self, **kwargs):
    super(ReduceSquareDifference, self).__init__(**kwargs)

  def _create_tensor(self):
    a = self.in_layers[0].out_tensor
    b = self.in_layers[1].out_tensor
    self.out_tensor = tf.reduce_mean(tf.squared_difference(a, b))
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


class AtomicConvolution(Layer):
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
    self.boxsize = boxsize
    self.radial_params = radial_params
    self.atom_types = atom_types
    super(AtomicConvolution, self).__init__(**kwargs)

  def _create_tensor(self):
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

    X = self.in_layers[0].out_tensor
    Nbrs = tf.to_int32(self.in_layers[1].out_tensor)
    Nbrs_Z = self.in_layers[2].out_tensor

    # N: Maximum number of atoms
    # M: Maximum number of neighbors
    # d: Number of coordinates/features/filters
    # B: Batch Size
    N = X.get_shape()[-2].value
    d = X.get_shape()[-1].value
    M = Nbrs.get_shape()[-1].value
    B = X.get_shape()[0].value

    D = self.distance_tensor(X, Nbrs, self.boxsize, B, N, M, d)
    R = self.distance_matrix(D)
    sym = []
    rsf_zeros = tf.zeros((B, N, M))
    for param in self.radial_params:

      # We apply the radial pooling filter before atom type conv
      # to reduce computation
      rsf = self.radial_symmetry_function(R, *param)

      if not self.atom_types:
        cond = tf.not_equal(Nbrs_Z, 0.0)
        sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))
      else:
        for j in range(len(self.atom_types)):
          cond = tf.equal(Nbrs_Z, self.atom_types[j])
          sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))

    layer = tf.stack(sym)
    layer = tf.transpose(layer, [1, 2, 0])
    m, v = tf.nn.moments(layer, axes=[0])
    self.out_tensor = tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)
    return self.out_tensor

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

    with tf.name_scope(None, "NbrRadialSymmetryFunction", [rc, rs, e]):
      rc = tf.Variable(rc)
      rs = tf.Variable(rs)
      e = tf.Variable(e)
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

    return tf.exp(-e * (R - rs) ** 2)

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
    atom_tensors = tf.unstack(X, axis=1)
    nbr_tensors = tf.unstack(Nbrs, axis=1)
    D = []
    if boxsize is not None:
      for atom, atom_tensor in enumerate(atom_tensors):
        nbrs = self.gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
        nbrs_tensors = tf.unstack(nbrs, axis=1)
        for nbr, nbr_tensor in enumerate(nbrs_tensors):
          _D = tf.subtract(nbr_tensor, atom_tensor)
          _D = tf.subtract(_D, boxsize * tf.round(tf.div(_D, boxsize)))
          D.append(_D)
    else:
      for atom, atom_tensor in enumerate(atom_tensors):
        nbrs = self.gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
        nbrs_tensors = tf.unstack(nbrs, axis=1)
        for nbr, nbr_tensor in enumerate(nbrs_tensors):
          _D = tf.subtract(nbr_tensor, atom_tensor)
          D.append(_D)
    D = tf.stack(D)
    D = tf.transpose(D, perm=[1, 0, 2])
    D = tf.reshape(D, [B, N, M, d])
    return D

  def gather_neighbors(self, X, nbr_indices, B, N, M, d):
    """Gathers the neighbor subsets of the atoms in X.
  
    B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features
  
    Parameters
    ----------
    X: tf.Tensor of shape (B, N, d)
      Coordinates/features tensor.
    atom_indices: tf.Tensor of shape (B, M)
      Neighbor list for single atom.
  
    Returns
    -------
    neighbors: tf.Tensor of shape (B, M, d)
      Neighbor coordinates/features tensor for single atom.
  
    """

    example_tensors = tf.unstack(X, axis=0)
    example_nbrs = tf.unstack(nbr_indices, axis=0)
    all_nbr_coords = []
    for example, (example_tensor,
                  example_nbr) in enumerate(zip(example_tensors, example_nbrs)):
      nbr_coords = tf.gather(example_tensor, example_nbr)
      all_nbr_coords.append(nbr_coords)
    neighbors = tf.stack(all_nbr_coords)
    return neighbors

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
