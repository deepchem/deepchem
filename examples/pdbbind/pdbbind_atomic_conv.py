"""
Script that trains Atomic Convs on PDBbind dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deepchem.models.tensorgraph.layers import Layer, Feature, Label, L2LossLayer
from deepchem.models import TensorGraph

import numpy as np
import tensorflow as tf
import itertools
import time

np.random.seed(123)
tf.set_random_seed(123)

import deepchem as dc
from pdbbind_datasets import load_pdbbind_grid

split = "random"
subset = "core"
tasks, datasets, _ = load_pdbbind_grid(split, featurizer="atomic_conv", subset=subset)
train_dataset, valid_dataset, test_dataset = datasets


def InitializeWeightsBiases(prev_layer_size,
                            size,
                            weights=None,
                            biases=None,
                            name=None):
  """Initializes weights and biases to be used in a fully-connected layer.

  Parameters
  ----------
  prev_layer_size: int
    Number of features in previous layer.
  size: int 
    Number of nodes in this layer.
  weights: tf.Tensor, optional (Default None)
    Weight tensor.
  biases: tf.Tensor, optional (Default None)
    Bias tensor.
  name: str 
    Name for this op, optional (Defaults to 'fully_connected' if None)

  Returns
  -------
  weights: tf.Variable
    Initialized weights.
  biases: tf.Variable
    Initialized biases.

  """

  if weights is None:
    weights = tf.truncated_normal([prev_layer_size, size], stddev=0.01)
  if biases is None:
    biases = tf.zeros([size])

  with tf.name_scope(name, 'fully_connected', [weights, biases]):
    w = tf.Variable(weights, name='w')
    b = tf.Variable(biases, name='b')
  return w, b


def AtomicNNLayer(tensor, size, weights, biases, name=None):
  """Fully connected layer with pre-initialized weights and biases.

  Parameters
  ----------
  tensor: tf.Tensor
    Input tensor.
  size: int
    Number of nodes in this layer.
  weights: tf.Variable 
    Initialized weights.
  biases: tf.Variable
    Initialized biases.
  name: str 
    Name for this op, optional (Defaults to 'fully_connected' if None)

  Returns
  -------
  retval: tf.Tensor
    A new tensor representing the output of the fully connected layer.

  Raises
  ------
  ValueError: If input tensor is not 2D.

  """

  if len(tensor.get_shape()) != 2:
    raise ValueError('Dense layer input must be 2D, not %dD' %
                     len(tensor.get_shape()))
  with tf.name_scope(name, 'fully_connected', [tensor, weights, biases]):
    return tf.nn.xw_plus_b(tensor, weights, biases)


### Atomicnet coordinate transform ops ###


def gather_neighbors(X, nbr_indices, B, N, M, d):
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


def DistanceTensor(X, Nbrs, boxsize, B, N, M, d):
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
  print(B, N, M, d)
  atom_tensors = tf.unstack(X, axis=1)
  nbr_tensors = tf.unstack(Nbrs, axis=1)
  D = []
  if boxsize is not None:
    for atom, atom_tensor in enumerate(atom_tensors):
      nbrs = gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
      nbrs_tensors = tf.unstack(nbrs, axis=1)
      for nbr, nbr_tensor in enumerate(nbrs_tensors):
        _D = tf.subtract(nbr_tensor, atom_tensor)
        _D = tf.subtract(_D, boxsize * tf.round(tf.div(_D, boxsize)))
        D.append(_D)
  else:
    for atom, atom_tensor in enumerate(atom_tensors):
      nbrs = gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
      nbrs_tensors = tf.unstack(nbrs, axis=1)
      for nbr, nbr_tensor in enumerate(nbrs_tensors):
        _D = tf.subtract(nbr_tensor, atom_tensor)
        D.append(_D)
  D = tf.stack(D)
  D = tf.transpose(D, perm=[1, 0, 2])
  D = tf.reshape(D, [B, N, M, d])
  return D


def DistanceMatrix(D):
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


### Atomicnet symmetry function kernel ops ###


def GaussianDistanceMatrix(R, rs, e):
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


def RadialCutoff(R, rc):
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


### Atomicnet symmetry function ops ###


def RadialSymmetryFunction(R, rc, rs, e):
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
    K = GaussianDistanceMatrix(R, rs, e)
    FC = RadialCutoff(R, rc)
  return tf.multiply(K, FC)


### Atomcnet symmetry function layer ops ###


def AtomicConvolutionLayer(X, Nbrs, Nbrs_Z, atom_types, radial_params, boxsize,
                           B, N, M, d):
  """Atomic convoluation layer

  N = max_num_atoms, M = max_num_neighbors, B = batch_size, d = num_features
  l = num_radial_filters * num_atom_types

  Parameters
  ----------
  X: tf.Tensor of shape (B, N, d)
    Coordinates/features.
  Nbrs: tf.Tensor of shape (B, N, M)
    Neighbor list.
  Nbrs_Z: tf.Tensor of shape (B, N, M)
    Atomic numbers of neighbor atoms.
  atom_types: list or None
    Of length a, where a is number of atom types for filtering.
  radial_params: list
    Of length l, where l is number of radial filters learned.
  boxsize: float or None
    Simulation box length [Angstrom].
  N: int
    Maximum number of atoms
  M: int
    Maximum number of neighbors
  d: int
    Number of coordinates/features/filters

  Returns
  -------
  layer: tf.Tensor of shape (l, B, N)
    A new tensor representing the output of the atomic conv layer 

  """

  D = DistanceTensor(X, Nbrs, boxsize, B, N, M, d)
  R = DistanceMatrix(D)
  sym = []
  rsf_zeros = tf.zeros((B, N, M))
  for param in radial_params:

    # We apply the radial pooling filter before atom type conv
    # to reduce computation
    rsf = RadialSymmetryFunction(R, *param)

    if not atom_types:
      cond = tf.not_equal(Nbrs_Z, 0.0)
      sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))
    else:
      for j in range(len(atom_types)):
        cond = tf.equal(Nbrs_Z, atom_types[j])
        sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))

  # Pack l (B, N) tensors into one (l, B, N) tensor
  # Transpose to (B, N, l) for conv layer stacking
  # done inside conv_layer loops to reduce transpose ops
  # Final layer should be shape (N, B, l) to pass into tf.map_fn
  layer = tf.stack(sym)
  return layer


class AtomicConv(Layer):
  def __init__(self, layer_sizes, weight_init_stddevs, bias_init_const,
               dropouts, boxsize, conv_layers, radial_params, atom_types
               , **kwargs):
    self.layer_sizes = layer_sizes
    self.weight_init_stddevs = weight_init_stddevs
    self.bias_init_consts = bias_init_const
    self.dropouts = dropouts
    self.boxsize = boxsize
    self.conv_layers = conv_layers
    self.radial_params = radial_params
    self.atom_types = atom_types
    super(AtomicConv, self).__init__(**kwargs)

  def _create_tensor(self):
    frag1_X_placeholder = self.in_layers[0].out_tensor
    frag1_Nbrs_placeholder = tf.to_int32(self.in_layers[1].out_tensor)
    frag1_Nbrs_Z_placeholder = self.in_layers[2].out_tensor
    frag2_X_placeholder = self.in_layers[3].out_tensor
    frag2_Nbrs_placeholder = tf.to_int32(self.in_layers[4].out_tensor)
    frag2_Nbrs_Z_placeholder = self.in_layers[5].out_tensor
    complex_X_placeholder = self.in_layers[6].out_tensor
    complex_Nbrs_placeholder = tf.to_int32(self.in_layers[7].out_tensor)
    complex_Nbrs_Z_placeholder = self.in_layers[8].out_tensor

    N = complex_X_placeholder.get_shape()[1].value
    N_1 = frag1_X_placeholder.get_shape()[1].value
    N_2 = frag2_X_placeholder.get_shape()[1].value
    M = frag1_Nbrs_placeholder.get_shape()[-1].value
    B = frag1_X_placeholder.get_shape()[0].value

    layer_sizes = self.layer_sizes
    weight_init_stddevs = self.weight_init_stddevs
    bias_init_consts = self.bias_init_consts
    dropouts = self.dropouts
    boxsize = self.boxsize
    conv_layers = self.conv_layers
    lengths_set = {
      len(layer_sizes),
      len(weight_init_stddevs),
      len(bias_init_consts),
      len(dropouts),
    }
    assert len(lengths_set) == 1, 'All layer params must have same length.'
    num_layers = lengths_set.pop()
    assert num_layers > 0, 'Must have some layers defined.'
    radial_params = self.radial_params
    atom_types = self.atom_types

    frag1_layer = AtomicConvolutionLayer(
      frag1_X_placeholder, frag1_Nbrs_placeholder,
      frag1_Nbrs_Z_placeholder, atom_types, radial_params, boxsize, B,
      N_1, M, 3)
    for x in range(conv_layers - 1):
      frag1_layer = tf.transpose(frag1_layer, [1, 2, 0])
      l = int(frag1_layer.get_shape()[-1])
      frag1_layer = AtomicConvolutionLayer(
        frag1_layer, frag1_Nbrs_placeholder,
        frag1_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
        B, N_1, M, l)

    frag2_layer = AtomicConvolutionLayer(
      frag2_X_placeholder, frag2_Nbrs_placeholder,
      frag2_Nbrs_Z_placeholder, atom_types, radial_params, boxsize, B,
      N_2, M, 3)
    for x in range(conv_layers - 1):
      frag2_layer = tf.transpose(frag2_layer, [1, 2, 0])
      l = int(frag2_layer.get_shape()[-1])
      frag2_layer = AtomicConvolutionLayer(
        frag2_layer, frag2_Nbrs_placeholder,
        frag2_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
        B, N_2, M, l)

    complex_layer = AtomicConvolutionLayer(
      complex_X_placeholder, complex_Nbrs_placeholder,
      complex_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
      B, N, M, 3)
    for x in range(conv_layers - 1):
      complex_layer = tf.transpose(complex_layer, [1, 2, 0])
      l = int(complex_layer.get_shape()[-1])
      complex_layer = AtomicConvolutionLayer(
        complex_layer, complex_Nbrs_placeholder,
        complex_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
        B, N, M, l)

    weights = []
    biases = []
    output_weights = []
    output_biases = []

    print("Atomic Conv Layer Built")
    frag1_layer = tf.transpose(frag1_layer, [2, 1, 0])
    frag2_layer = tf.transpose(frag2_layer, [2, 1, 0])
    complex_layer = tf.transpose(complex_layer, [2, 1, 0])
    prev_layer_size = int(tf.Tensor.get_shape(complex_layer)[2])
    for i in range(num_layers):
      weight, bias = InitializeWeightsBiases(
        prev_layer_size=prev_layer_size,
        size=layer_sizes[i],
        weights=tf.truncated_normal(
          shape=[prev_layer_size, layer_sizes[i]],
          stddev=weight_init_stddevs[i]),
        biases=tf.constant(
          value=bias_init_consts[i], shape=[layer_sizes[i]]))
      weights.append(weight)
      biases.append(bias)
      prev_layer_size = layer_sizes[i]
    weight, bias = InitializeWeightsBiases(prev_layer_size, 1)
    output_weights.append(weight)
    output_biases.append(bias)

    def atomnet(current_input):
      prev_layer = current_input
      for i in range(num_layers):
        layer = AtomicNNLayer(
          tensor=prev_layer,
          size=layer_sizes[i],
          weights=weights[i],
          biases=biases[i])
        layer = tf.nn.relu(layer)
        # TODO (LESWING)
        # layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
      output_layer = tf.squeeze(
        AtomicNNLayer(
          tensor=prev_layer,
          size=prev_layer_size,
          weights=output_weights[0],
          biases=output_biases[0]))
      return output_layer
    print("Per Atom Dense Built")

    frag1_outputs = tf.map_fn(lambda x: atomnet(x), frag1_layer)
    frag2_outputs = tf.map_fn(lambda x: atomnet(x), frag2_layer)
    complex_outputs = tf.map_fn(lambda x: atomnet(x), complex_layer)
    frag1_energy = tf.reduce_sum(frag1_outputs, 0)
    frag2_energy = tf.reduce_sum(frag2_outputs, 0)
    complex_energy = tf.reduce_sum(complex_outputs, 0)
    binding_energy = complex_energy - frag1_energy - frag2_energy
    self.out_tensor = tf.expand_dims(binding_energy, axis=1)
    return self.out_tensor


transformers = [
  dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
]

for transformer in transformers:
  train_dataset = transformer.transform(train_dataset)
  test_dataset = transformer.transform(test_dataset)

frag1_num_atoms = 140
frag2_num_atoms = 821
complex_num_atoms = 908
max_num_neighbors = 12
neighbor_cutoff = 12.0
batch_size = 80

at = [1., 6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]
radial = [[12.0], [0.0, 4.0, 8.0], [4.0]]
rp = [x for x in itertools.product(*radial)]
layer_sizes = [32, 32, 16]
weight_init_stddevs = [
  1 / np.sqrt(layer_sizes[0]), 1 / np.sqrt(layer_sizes[1]),
  1 / np.sqrt(layer_sizes[2])
]
bias_init_consts = [0., 0., 0.]
dropouts = [0., 0., 0.]
penalty_type = "l2"
penalty = 0.

frag1_X = Feature(shape=(batch_size, frag1_num_atoms, 3))
frag1_nbrs = Feature(shape=(batch_size, frag1_num_atoms, max_num_neighbors))
frag1_nbrs_z = Feature(shape=(batch_size, frag1_num_atoms, max_num_neighbors))

frag2_X = Feature(shape=(batch_size, frag2_num_atoms, 3))
frag2_nbrs = Feature(shape=(batch_size, frag2_num_atoms, max_num_neighbors))
frag2_nbrs_z = Feature(shape=(batch_size, frag2_num_atoms, max_num_neighbors))

complex_X = Feature(shape=(batch_size, complex_num_atoms, 3))
complex_nbrs = Feature(shape=(batch_size, complex_num_atoms, max_num_neighbors))
complex_nbrs_z = Feature(shape=(batch_size, complex_num_atoms, max_num_neighbors))

conv_layer = AtomicConv(layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
                        boxsize=None, conv_layers=1, radial_params=rp, atom_types=at,
                        in_layers=[frag1_X, frag1_nbrs, frag1_nbrs_z, frag2_X,
                                   frag2_nbrs, frag2_nbrs_z, complex_X,
                                   complex_nbrs, complex_nbrs_z])

label = Label(shape=(None, 1))
loss = L2LossLayer(in_layers=[conv_layer, label])


def feed_dict_generator(dataset, batch_size, epochs=1):
  total_time = 0
  for epoch in range(epochs):
    for ind, (F_b, y_b, w_b, ids_b) in enumerate(
      dataset.iterbatches(batch_size, pad_batches=True)):
      time1 = time.time()
      N = complex_num_atoms
      N_1 = frag1_num_atoms
      N_2 = frag2_num_atoms
      M = max_num_neighbors

      orig_dict = {}
      batch_size = F_b.shape[0]
      num_features = F_b[0][0].shape[1]
      frag1_X_b = np.zeros((batch_size, N_1, num_features))
      for i in range(batch_size):
        frag1_X_b[i] = F_b[i][0]
      orig_dict[frag1_X] = frag1_X_b

      frag2_X_b = np.zeros((batch_size, N_2, num_features))
      for i in range(batch_size):
        frag2_X_b[i] = F_b[i][3]
      orig_dict[frag2_X] = frag2_X_b

      complex_X_b = np.zeros((batch_size, N, num_features))
      for i in range(batch_size):
        complex_X_b[i] = F_b[i][6]
      orig_dict[complex_X] = complex_X_b

      frag1_Nbrs = np.zeros((batch_size, N_1, M))
      frag1_Z_b = np.zeros((batch_size, N_1))
      for i in range(batch_size):
        frag1_Z_b[i] = F_b[i][2]
      frag1_Nbrs_Z = np.zeros((batch_size, N_1, M))
      for atom in range(N_1):
        for i in range(batch_size):
          atom_nbrs = F_b[i][1].get(atom, "")
          frag1_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
          for j, atom_j in enumerate(atom_nbrs):
            frag1_Nbrs_Z[i, atom, j] = frag1_Z_b[i, atom_j]

      orig_dict[frag1_nbrs] = frag1_Nbrs
      orig_dict[frag1_nbrs_z] = frag1_Nbrs_Z

      frag2_Nbrs = np.zeros((batch_size, N_2, M))
      frag2_Z_b = np.zeros((batch_size, N_2))
      for i in range(batch_size):
        frag2_Z_b[i] = F_b[i][5]
      frag2_Nbrs_Z = np.zeros((batch_size, N_2, M))
      for atom in range(N_2):
        for i in range(batch_size):
          atom_nbrs = F_b[i][4].get(atom, "")
          frag2_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
          for j, atom_j in enumerate(atom_nbrs):
            frag2_Nbrs_Z[i, atom, j] = frag2_Z_b[i, atom_j]

      orig_dict[frag2_nbrs] = frag2_Nbrs
      orig_dict[frag2_nbrs_z] = frag2_Nbrs_Z

      complex_Nbrs = np.zeros((batch_size, N, M))
      complex_Z_b = np.zeros((batch_size, N))
      for i in range(batch_size):
        complex_Z_b[i] = F_b[i][8]
      complex_Nbrs_Z = np.zeros((batch_size, N, M))
      for atom in range(N):
        for i in range(batch_size):
          atom_nbrs = F_b[i][7].get(atom, "")
          complex_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
          for j, atom_j in enumerate(atom_nbrs):
            complex_Nbrs_Z[i, atom, j] = complex_Z_b[i, atom_j]

      orig_dict[complex_nbrs] = complex_Nbrs
      orig_dict[complex_nbrs_z] = complex_Nbrs_Z
      orig_dict[label] = np.reshape(y_b, newshape=(batch_size, 1))
      time2 = time.time()
      total_time += time1 - time2
      # print("total time: %s" % total_time)
      yield orig_dict


tg = TensorGraph(batch_size=batch_size,
                 mode=str("regression"),
                 model_dir=str("/tmp/atom_conv"))
tg.add_output(conv_layer)
tg.set_loss(loss)

print("Fitting")
tg.fit_generator(feed_dict_generator(train_dataset, batch_size, epochs=1))

metric = [
  dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
  dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]
train_evaluator = dc.utils.evaluate.GeneratorEvaluator(tg, feed_dict_generator(train_dataset, batch_size),
                                                       transformers, [label])
train_scores = train_evaluator.compute_model_performance(metric)
print("Train scores")
print(train_scores)
test_evaluator = dc.utils.evaluate.GeneratorEvaluator(tg, feed_dict_generator(test_dataset, batch_size),
                                                      transformers, [label])
test_scores = test_evaluator.compute_model_performance(metric)
print("Test scores")
print(test_scores)
