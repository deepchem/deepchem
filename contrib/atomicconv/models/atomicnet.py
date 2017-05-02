from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import time
import numpy as np
import tensorflow as tf

from deepchem.data import DiskDataset
from deepchem.nn import model_ops
from legacy import TensorflowGraph, TensorflowGraphModel, TensorflowMultiTaskRegressor
from deepchem.utils.save import log
import atomicnet_ops


class TensorflowFragmentRegressor(TensorflowMultiTaskRegressor):
  """Create atomic convolution neural network potential for binding energy.

  Example:

  >>> B = 10 # batch_size
  >>> N_1 = 6 # frag1_n_atoms
  >>> N_2 = 6 # frag2_n_atoms
  >>> N = 12 # complex_n_atoms
  >>> M = 6 # n_neighbors
  >>> n_tasks = 1
  >>> C_1 = np.zeros((N_1, 3))
  >>> C_2 = np.zeros((N_2, 3))
  >>> C = np.zeros((N, 3))
  >>> NL_1 = {}
  >>> for i in range(n_atoms): NL_1[i] = [0 for m in range(M)]
  >>> NL_2 = {}
  >>> for i in range(n_atoms): NL_2[i] = [0 for m in range(M)]
  >>> NL = {}
  >>> for i in range(n_atoms): NL[i] = [0 for m in range(M)]
  >>> Z_1 = np.zeros((N))
  >>> Z_2 = np.zeros((N))
  >>> Z = np.zeros((N))
  >>> X = [(C_1, NL_1, Z_1, C_2, NL_2, Z_2, C, NL, Z) for i in range(B)]
  >>> y = np.zeros(B, n_tasks)
  >>> w = np.zeros(B, n_tasks)
  >>> ids = np.zeros(B,)
  >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
  >>> rp = [[12.0, 0.0, 0.04]]
  >>> at = None
  >>> model = TensorflowFragmentRegressor(n_tasks, rp, at, N_1, N_2, N, M)
  >>> model.fit(dataset)

  """

  def __init__(self,
               n_tasks,
               radial_params,
               atom_types,
               frag1_num_atoms,
               frag2_num_atoms,
               complex_num_atoms,
               max_num_neighbors,
               logdir=None,
               layer_sizes=[100],
               weight_init_stddevs=[0.1],
               bias_init_consts=[1.],
               penalty=0.0,
               penalty_type="l2",
               dropouts=[0.5],
               learning_rate=.001,
               momentum=.8,
               optimizer="adam",
               batch_size=48,
               conv_layers=1,
               boxsize=None,
               verbose=True,
               seed=None):
    """Initialize TensorflowFragmentRegressor.

    Parameters
    ----------

    n_tasks: int
      Number of tasks.
    radial_params: list
      Of length l, where l is number of radial filters learned.
    atom_types: list
      Of length a, where a is number of atom_types for filtering.
    frag1_num_atoms: int
      Maximum number of atoms in fragment 1.
    frag2_num_atoms: int
      Maximum number of atoms in fragment 2.
    complex_num_atoms: int
      Maximum number of atoms in complex.
    max_num_neighbors: int
      Maximum number of neighbors per atom.
    logdir: str
      Path to model save directory.
    layer_sizes: list
      List of layer sizes.
    weight_init_stddevs: list
      List of standard deviations for weights (sampled from zero-mean
      gaussians). One for each layer.
    bias_init_consts: list
      List of bias initializations. One for each layer.
    penalty: float
      Amount of penalty (l2 or l1 applied)
    penalty_type: str
      Either "l2" or "l1"
    dropouts: list
      List of dropout amounts. One for each layer.
    learning_rate: float
      Learning rate for model.
    momentum: float
      Momentum. Only applied if optimizer=="momentum"
    optimizer: str
      Type of optimizer applied.
    batch_size: int
      Size of minibatches for training.
    conv_layers: int
      Number of atomic convolution layers (experimental feature).
    boxsize: float or None
      Simulation box length [Angstrom]. If None, no periodic boundary conditions.
    verbose: bool, optional (Default True)
      Whether to perform logging.
    seed: int, optional (Default None)
      If not none, is used as random seed for tensorflow.

    """

    self.n_tasks = n_tasks
    self.radial_params = radial_params
    self.atom_types = atom_types
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.complex_num_atoms = complex_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.conv_layers = conv_layers
    self.boxsize = boxsize
    TensorflowGraphModel.__init__(
        self,
        n_tasks,
        None,
        logdir,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        penalty=penalty,
        penalty_type=penalty_type,
        dropouts=dropouts,
        learning_rate=learning_rate,
        momentum=momentum,
        optimizer=optimizer,
        batch_size=batch_size,
        pad_batches=True,
        verbose=verbose,
        seed=seed)

  def construct_feed_dict(self, F_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    B = batch_size, N = max_num_atoms

    Parameters
    ----------

    F_b: np.ndarray of B tuples of (X_1, L_1, Z_1, X_2, L_2, Z_2, X, L, Z) 
      X_1: ndarray shape (N, 3).
        Fragment 1 Cartesian coordinates [Angstrom].
      L_1: dict with N keys.
        Fragment 1 neighbor list.
      Z_1: ndarray shape (N,).
        Fragment 1 atomic numbers.
      X_2: ndarray shape (N, 3).
        Fragment 2 Cartesian coordinates [Angstrom].
      L_2: dict with N keys.
        Fragment 2 neighbor list.
      Z_2: ndarray shape (N,).
        Fragment 2 atomic numbers.
      X: ndarray shape (N, 3).
        Complex Cartesian coordinates [Angstrom].
      L: dict with N keys.
        Complex neighbor list.
      Z: ndarray shape (N,).
        Complex atomic numbers.
    y_b: np.ndarray of shape (B, num_tasks)
      Tasks.
    w_b: np.ndarray of shape (B, num_tasks)
      Task weights.
    ids_b: List of length (B,) 
      Datapoint identifiers. Not currently used.

    Returns
    -------

    retval: dict
      Tensorflow feed dict

    """

    N = self.complex_num_atoms
    N_1 = self.frag1_num_atoms
    N_2 = self.frag2_num_atoms
    M = self.max_num_neighbors

    orig_dict = {}
    batch_size = F_b.shape[0]
    num_features = F_b[0][0].shape[1]
    frag1_X_b = np.zeros((batch_size, N_1, num_features))
    for i in range(batch_size):
      frag1_X_b[i] = F_b[i][0]
    orig_dict["frag1_X_placeholder"] = frag1_X_b

    frag2_X_b = np.zeros((batch_size, N_2, num_features))
    for i in range(batch_size):
      frag2_X_b[i] = F_b[i][3]
    orig_dict["frag2_X_placeholder"] = frag2_X_b

    complex_X_b = np.zeros((batch_size, N, num_features))
    for i in range(batch_size):
      complex_X_b[i] = F_b[i][6]
    orig_dict["complex_X_placeholder"] = complex_X_b

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

    orig_dict["frag1_Z_placeholder"] = frag1_Z_b
    orig_dict["frag1_Nbrs_placeholder"] = frag1_Nbrs
    orig_dict["frag1_Nbrs_Z_placeholder"] = frag1_Nbrs_Z

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

    orig_dict["frag2_Z_placeholder"] = frag2_Z_b
    orig_dict["frag2_Nbrs_placeholder"] = frag2_Nbrs
    orig_dict["frag2_Nbrs_Z_placeholder"] = frag2_Nbrs_Z

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

    orig_dict["complex_Z_placeholder"] = complex_Z_b
    orig_dict["complex_Nbrs_placeholder"] = complex_Nbrs
    orig_dict["complex_Nbrs_Z_placeholder"] = complex_Nbrs_Z
    for task in range(self.n_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = y_b[:, task]
      else:
        orig_dict["labels_%d" % task] = np.zeros((self.batch_size,))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        orig_dict["weights_%d" % task] = np.ones((self.batch_size,))
    return TensorflowGraph.get_feed_dict(orig_dict)

  def build(self, graph, name_scopes, training):

    N = self.complex_num_atoms
    N_1 = self.frag1_num_atoms
    N_2 = self.frag2_num_atoms
    M = self.max_num_neighbors
    B = self.batch_size
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph,
                                                              name_scopes)
    with graph.as_default():
      with placeholder_scope:
        self.frag1_X_placeholder = tf.placeholder(
            tf.float32, shape=[B, N_1, 3], name='frag1_X_placeholder')
        self.frag1_Z_placeholder = tf.placeholder(
            tf.float32, shape=[B, N_1], name='frag1_Z_placeholder')
        self.frag1_Nbrs_placeholder = tf.placeholder(
            tf.int32, shape=[B, N_1, M], name="frag1_Nbrs_placeholder")
        self.frag1_Nbrs_Z_placeholder = tf.placeholder(
            tf.float32, shape=[B, N_1, M], name='frag1_Nbrs_Z_placeholder')
        self.frag2_X_placeholder = tf.placeholder(
            tf.float32, shape=[B, N_2, 3], name='frag2_X_placeholder')
        self.frag2_Z_placeholder = tf.placeholder(
            tf.float32, shape=[B, N_2], name='frag2_Z_placeholder')
        self.frag2_Nbrs_placeholder = tf.placeholder(
            tf.int32, shape=[B, N_2, M], name="frag2_Nbrs_placeholder")
        self.frag2_Nbrs_Z_placeholder = tf.placeholder(
            tf.float32, shape=[B, N_2, M], name='frag2_Nbrs_Z_placeholder')
        self.complex_X_placeholder = tf.placeholder(
            tf.float32, shape=[B, N, 3], name='complex_X_placeholder')
        self.complex_Z_placeholder = tf.placeholder(
            tf.float32, shape=[B, N], name='complex_Z_placeholder')
        self.complex_Nbrs_placeholder = tf.placeholder(
            tf.int32, shape=[B, N, M], name="complex_Nbrs_placeholder")
        self.complex_Nbrs_Z_placeholder = tf.placeholder(
            tf.float32, shape=[B, N, M], name='complex_Nbrs_Z_placeholder')

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

      frag1_layer = atomicnet_ops.AtomicConvolutionLayer(
          self.frag1_X_placeholder, self.frag1_Nbrs_placeholder,
          self.frag1_Nbrs_Z_placeholder, atom_types, radial_params, boxsize, B,
          N_1, M, 3)
      for x in range(conv_layers - 1):
        l = int(frag1_layer.get_shape()[-1])
        frag1_layer = atomicnet_ops.AtomicConvolutionLayer(
            frag1_layer, self.frag1_Nbrs_placeholder,
            self.frag1_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
            B, N_1, M, l)

      frag2_layer = atomicnet_ops.AtomicConvolutionLayer(
          self.frag2_X_placeholder, self.frag2_Nbrs_placeholder,
          self.frag2_Nbrs_Z_placeholder, atom_types, radial_params, boxsize, B,
          N_2, M, 3)
      for x in range(conv_layers - 1):
        l = int(frag2_layer.get_shape()[-1])
        frag2_layer = atomicnet_ops.AtomicConvolutionLayer(
            frag2_layer, self.frag2_Nbrs_placeholder,
            self.frag2_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
            B, N_2, M, l)

      complex_layer = atomicnet_ops.AtomicConvolutionLayer(
          self.complex_X_placeholder, self.complex_Nbrs_placeholder,
          self.complex_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
          B, N, M, 3)
      for x in range(conv_layers - 1):
        l = int(complex_layer.get_shape()[-1])
        complex_layer = atomicnet_ops.AtomicConvolutionLayer(
            complex_layer, self.complex_Nbrs_placeholder,
            self.complex_Nbrs_Z_placeholder, atom_types, radial_params, boxsize,
            B, N, M, l)

      weights = []
      biases = []
      output_weights = []
      output_biases = []
      output = []

      n_features = int(frag1_layer.get_shape()[-1])

      for ind, atomtype in enumerate(atom_types):

        prev_layer_size = n_features
        weights.append([])
        biases.append([])
        output_weights.append([])
        output_biases.append([])
        for i in range(num_layers):
          weight, bias = atomicnet_ops.InitializeWeightsBiases(
              prev_layer_size=prev_layer_size,
              size=layer_sizes[i],
              weights=tf.truncated_normal(
                  shape=[prev_layer_size, layer_sizes[i]],
                  stddev=weight_init_stddevs[i]),
              biases=tf.constant(
                  value=bias_init_consts[i], shape=[layer_sizes[i]]))
          weights[ind].append(weight)
          biases[ind].append(bias)
          prev_layer_size = layer_sizes[i]
        weight, bias = atomicnet_ops.InitializeWeightsBiases(prev_layer_size, 1)
        output_weights[ind].append(weight)
        output_biases[ind].append(bias)

      def atomnet(current_input, atomtype):

        prev_layer = current_input

        for i in range(num_layers):
          layer = atomicnet_ops.AtomicNNLayer(
              tensor=prev_layer,
              size=layer_sizes[i],
              weights=weights[atomtype][i],
              biases=biases[atomtype][i])
          layer = tf.nn.relu(layer)
          layer = model_ops.dropout(layer, dropouts[i], training)
          prev_layer = layer
          prev_layer_size = layer_sizes[i]

        output_layer = tf.squeeze(
            atomicnet_ops.AtomicNNLayer(
                tensor=prev_layer,
                size=prev_layer_size,
                weights=output_weights[atomtype][0],
                biases=output_biases[atomtype][0]))

        return output_layer

      frag1_zeros = tf.zeros((B, N_1))
      frag2_zeros = tf.zeros((B, N_2))
      complex_zeros = tf.zeros((B, N))

      frag1_atomtype_energy = []
      frag2_atomtype_energy = []
      complex_atomtype_energy = []

      for ind, atomtype in enumerate(atom_types):

        frag1_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag1_layer)
        frag2_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag2_layer)
        complex_outputs = tf.map_fn(lambda x: atomnet(x, ind), complex_layer)

        cond = tf.equal(self.frag1_Z_placeholder, atomtype)
        frag1_atomtype_energy.append(tf.where(cond, frag1_outputs, frag1_zeros))
        cond = tf.equal(self.frag2_Z_placeholder, atomtype)
        frag2_atomtype_energy.append(tf.where(cond, frag2_outputs, frag2_zeros))
        cond = tf.equal(self.complex_Z_placeholder, atomtype)
        complex_atomtype_energy.append(
            tf.where(cond, complex_outputs, complex_zeros))

      frag1_outputs = tf.add_n(frag1_atomtype_energy)
      frag2_outputs = tf.add_n(frag2_atomtype_energy)
      complex_outputs = tf.add_n(complex_atomtype_energy)

      frag1_energy = tf.reduce_sum(frag1_outputs, 1)
      frag2_energy = tf.reduce_sum(frag2_outputs, 1)
      complex_energy = tf.reduce_sum(complex_outputs, 1)
      binding_energy = complex_energy - (frag1_energy + frag2_energy)
      output.append(binding_energy)

    return output
