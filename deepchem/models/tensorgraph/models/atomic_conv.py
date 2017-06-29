from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import sys

sys.path.append("../../models")
from deepchem.models.tensorgraph.layers import Layer, Feature, Label, AtomicConvolution, L2Loss, ReduceMean
from deepchem.models import TensorGraph

import numpy as np
import tensorflow as tf
import itertools


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


class AtomicConvScore(Layer):

  def __init__(self, atom_types, layer_sizes, **kwargs):
    self.atom_types = atom_types
    self.layer_sizes = layer_sizes
    super(AtomicConvScore, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    frag1_layer = self.in_layers[0].out_tensor
    frag2_layer = self.in_layers[1].out_tensor
    complex_layer = self.in_layers[2].out_tensor

    frag1_z = self.in_layers[3].out_tensor
    frag2_z = self.in_layers[4].out_tensor
    complex_z = self.in_layers[5].out_tensor

    atom_types = self.atom_types
    layer_sizes = self.layer_sizes
    num_layers = len(layer_sizes)
    weight_init_stddevs = [1 / np.sqrt(x) for x in layer_sizes]
    bias_init_consts = [0.0] * num_layers

    weights = []
    biases = []
    output_weights = []
    output_biases = []

    n_features = int(frag1_layer.get_shape()[-1])

    for ind, atomtype in enumerate(atom_types):

      prev_layer_size = n_features
      weights.append([])
      biases.append([])
      output_weights.append([])
      output_biases.append([])
      for i in range(num_layers):
        weight, bias = InitializeWeightsBiases(
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
      weight, bias = InitializeWeightsBiases(prev_layer_size, 1)
      output_weights[ind].append(weight)
      output_biases[ind].append(bias)

    def atomnet(current_input, atomtype):
      prev_layer = current_input
      for i in range(num_layers):
        layer = tf.nn.xw_plus_b(prev_layer, weights[atomtype][i],
                                biases[atomtype][i])
        layer = tf.nn.relu(layer)
        prev_layer = layer

      output_layer = tf.squeeze(
          tf.nn.xw_plus_b(prev_layer, output_weights[atomtype][0],
                          output_biases[atomtype][0]))
      return output_layer

    frag1_zeros = tf.zeros_like(frag1_z, dtype=tf.float32)
    frag2_zeros = tf.zeros_like(frag2_z, dtype=tf.float32)
    complex_zeros = tf.zeros_like(complex_z, dtype=tf.float32)

    frag1_atomtype_energy = []
    frag2_atomtype_energy = []
    complex_atomtype_energy = []

    for ind, atomtype in enumerate(atom_types):
      frag1_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag1_layer)
      frag2_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag2_layer)
      complex_outputs = tf.map_fn(lambda x: atomnet(x, ind), complex_layer)

      cond = tf.equal(frag1_z, atomtype)
      frag1_atomtype_energy.append(tf.where(cond, frag1_outputs, frag1_zeros))
      cond = tf.equal(frag2_z, atomtype)
      frag2_atomtype_energy.append(tf.where(cond, frag2_outputs, frag2_zeros))
      cond = tf.equal(complex_z, atomtype)
      complex_atomtype_energy.append(
          tf.where(cond, complex_outputs, complex_zeros))

    frag1_outputs = tf.add_n(frag1_atomtype_energy)
    frag2_outputs = tf.add_n(frag2_atomtype_energy)
    complex_outputs = tf.add_n(complex_atomtype_energy)

    frag1_energy = tf.reduce_sum(frag1_outputs, 1)
    frag2_energy = tf.reduce_sum(frag2_outputs, 1)
    complex_energy = tf.reduce_sum(complex_outputs, 1)
    binding_energy = complex_energy - (frag1_energy + frag2_energy)
    self.out_tensor = tf.expand_dims(binding_energy, axis=1)
    return self.out_tensor


def atomic_conv_model(
    frag1_num_atoms=70,
    frag2_num_atoms=634,
    complex_num_atoms=701,
    max_num_neighbors=12,
    batch_size=24,
    at=[6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53., -1.],
    radial=[[
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
        8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
    ], [0.0, 4.0, 8.0], [0.4]],
    layer_sizes=[32, 32, 16],
    learning_rate=0.001):
  rp = [x for x in itertools.product(*radial)]
  frag1_X = Feature(shape=(batch_size, frag1_num_atoms, 3))
  frag1_nbrs = Feature(shape=(batch_size, frag1_num_atoms, max_num_neighbors))
  frag1_nbrs_z = Feature(shape=(batch_size, frag1_num_atoms, max_num_neighbors))
  frag1_z = Feature(shape=(batch_size, frag1_num_atoms))

  frag2_X = Feature(shape=(batch_size, frag2_num_atoms, 3))
  frag2_nbrs = Feature(shape=(batch_size, frag2_num_atoms, max_num_neighbors))
  frag2_nbrs_z = Feature(shape=(batch_size, frag2_num_atoms, max_num_neighbors))
  frag2_z = Feature(shape=(batch_size, frag2_num_atoms))

  complex_X = Feature(shape=(batch_size, complex_num_atoms, 3))
  complex_nbrs = Feature(shape=(batch_size, complex_num_atoms,
                                max_num_neighbors))
  complex_nbrs_z = Feature(shape=(batch_size, complex_num_atoms,
                                  max_num_neighbors))
  complex_z = Feature(shape=(batch_size, complex_num_atoms))

  frag1_conv = AtomicConvolution(
      atom_types=at,
      radial_params=rp,
      boxsize=None,
      in_layers=[frag1_X, frag1_nbrs, frag1_nbrs_z])

  frag2_conv = AtomicConvolution(
      atom_types=at,
      radial_params=rp,
      boxsize=None,
      in_layers=[frag2_X, frag2_nbrs, frag2_nbrs_z])

  complex_conv = AtomicConvolution(
      atom_types=at,
      radial_params=rp,
      boxsize=None,
      in_layers=[complex_X, complex_nbrs, complex_nbrs_z])

  score = AtomicConvScore(
      at,
      layer_sizes,
      in_layers=[
          frag1_conv, frag2_conv, complex_conv, frag1_z, frag2_z, complex_z
      ])

  label = Label(shape=(None, 1))
  loss = ReduceMean(in_layers=L2Loss(in_layers=[score, label]))

  def feed_dict_generator(dataset, batch_size, epochs=1, pad_batches=True):

    def replace_atom_types(z):

      def place_holder(i):
        if i in at:
          return i
        return -1

      return np.array([place_holder(x) for x in z])

    for epoch in range(epochs):
      for ind, (F_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(
              batch_size, deterministic=True, pad_batches=pad_batches)):
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
          z = replace_atom_types(F_b[i][2])
          frag1_Z_b[i] = z
        frag1_Nbrs_Z = np.zeros((batch_size, N_1, M))
        for atom in range(N_1):
          for i in range(batch_size):
            atom_nbrs = F_b[i][1].get(atom, "")
            frag1_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag1_Nbrs_Z[i, atom, j] = frag1_Z_b[i, atom_j]
        orig_dict[frag1_nbrs] = frag1_Nbrs
        orig_dict[frag1_nbrs_z] = frag1_Nbrs_Z
        orig_dict[frag1_z] = frag1_Z_b

        frag2_Nbrs = np.zeros((batch_size, N_2, M))
        frag2_Z_b = np.zeros((batch_size, N_2))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][5])
          frag2_Z_b[i] = z
        frag2_Nbrs_Z = np.zeros((batch_size, N_2, M))
        for atom in range(N_2):
          for i in range(batch_size):
            atom_nbrs = F_b[i][4].get(atom, "")
            frag2_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag2_Nbrs_Z[i, atom, j] = frag2_Z_b[i, atom_j]
        orig_dict[frag2_nbrs] = frag2_Nbrs
        orig_dict[frag2_nbrs_z] = frag2_Nbrs_Z
        orig_dict[frag2_z] = frag2_Z_b

        complex_Nbrs = np.zeros((batch_size, N, M))
        complex_Z_b = np.zeros((batch_size, N))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][8])
          complex_Z_b[i] = z
        complex_Nbrs_Z = np.zeros((batch_size, N, M))
        for atom in range(N):
          for i in range(batch_size):
            atom_nbrs = F_b[i][7].get(atom, "")
            complex_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              complex_Nbrs_Z[i, atom, j] = complex_Z_b[i, atom_j]

        orig_dict[complex_nbrs] = complex_Nbrs
        orig_dict[complex_nbrs_z] = complex_Nbrs_Z
        orig_dict[complex_z] = complex_Z_b
        orig_dict[label] = np.reshape(y_b, newshape=(batch_size, 1))
        yield orig_dict

  tg = TensorGraph(
      batch_size=batch_size,
      mode=str("regression"),
      model_dir=str("/tmp/atom_conv"),
      learning_rate=learning_rate)
  tg.add_output(score)
  tg.set_loss(loss)
  return tg, feed_dict_generator, label
