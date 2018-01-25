"""
Script that trains multitask models on Tox21 dataset.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

import deepchem as dc
from deepchem.molnet import load_tox21
from feat.mol_graphs import ConvMol
from tensorflow.python.keras import backend as K

sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(True)


def reshape_y(y):
  vectors = []
  for i in range(y.shape[1]):
    vectors.append(y[:, i])
  return vectors


# Only for debug!
np.random.seed(123)

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='GraphConv')
# tox21_tasks, tox21_datasets, transformers = load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)


class GraphConv(tf.keras.layers.Layer):

  def __init__(self, output_dim, activation_fn=None, **kwargs):
    self.min_degree = 0
    self.max_degree = 10
    self.num_deg = 2 * 10 + (1 - 0)
    self.activation_fn = tf.nn.relu
    self.out_channel = output_dim
    self.activation_fn = activation_fn
    super(GraphConv, self).__init__(**kwargs)

  def build(self, input_shape):
    self.W_list = [
        self.add_weight(
            name='w%s' % k,
            shape=(input_shape[0][-1], self.out_channel),
            initializer=tf.glorot_uniform_initializer(),
            trainable=True,
            dtype=tf.float32) for k in range(self.num_deg)
    ]

    self.b_list = [
        self.add_weight(
            name='b%s' % k,
            shape=(self.out_channel,),
            initializer=tf.zeros_initializer(),
            trainable=True,
            dtype=tf.float32) for k in range(self.num_deg)
    ]
    super(GraphConv, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, x, **kwargs):
    atom_features = x[0]

    # Extract graph topology
    deg_slice = x[1]
    deg_adj_lists = x[2:]

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
    return atom_features

  def compute_output_shape(self, input_shape):
    return (input_shape[0][0], self.out_channel)

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


class GraphGather(tf.keras.layers.Layer):

  def __init__(self, batch_size, activation_fn, **kwargs):
    self.batch_size = batch_size
    self.activation_fn = activation_fn
    super(GraphGather, self).__init__(**kwargs)

  def call(self, x, **kwargs):
    atom_features = x[0]

    # Extract graph topology
    membership = x[1]
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
    out_tensor = mol_features
    return out_tensor

  def compute_output_shape(self, input_shape):
    return (self.batch_size, 2 * input_shape[0][-1])


batch_size = 64
atom_features = tf.placeholder(
    tf.float32, shape=(
        None,
        75,
    ))
degree_slice = tf.placeholder(tf.int32, shape=(None, 2))
membership = tf.placeholder(tf.int32, shape=(None,))

deg_adjs = []
for i in range(10):
  deg_adj = tf.placeholder(
      tf.int32, shape=(
          None,
          i + 1,
      ))
  deg_adjs.append(deg_adj)

inputs = [atom_features, degree_slice, membership] + deg_adjs

x = GraphConv(
    64, activation_fn=tf.nn.relu)([atom_features, degree_slice] + deg_adjs)
# TODO(LESWING) GraphPool Conversion
x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.BatchNormalization()(x)
readout = GraphGather(
    batch_size=batch_size, activation_fn=tf.nn.tanh)([x, membership])

labels = []
weights = []
losses = []
outputs = []
for i in range(len(tox21_tasks)):
  logit = tf.keras.layers.Dense(2, activation=None)(readout)

  label = tf.placeholder(tf.int32, shape=(None))
  labels.append(label)

  weight = tf.placeholder(tf.float32, shape=(None,))
  weights.append(weight)

  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=label, logits=logit) * weight
  losses.append(loss)

  output = tf.nn.softmax(logit)
  outputs.append(output)

final_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer().minimize(final_loss)

init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
  for epoch in range(10):
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(
        train_dataset.iterbatches(
            batch_size, pad_batches=True, deterministic=False)):
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      X = []
      X.append(multiConvMol.get_atom_features())
      X.append(multiConvMol.deg_slice)
      X.append(np.array(multiConvMol.membership))
      for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        X.append(multiConvMol.get_deg_adjacency_lists()[i])
      y = reshape_y(y_b)
      w = reshape_y(w_b)
      f_dict = {x[0]: x[1] for x in zip(inputs, X)}
      f_dict.update({x[0]: x[1] for x in zip(labels, y)})
      f_dict.update({x[0]: x[1] for x in zip(weights, w)})
      sess.run(train_step, feed_dict=f_dict)
