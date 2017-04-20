from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import os
import sys
import deepchem as dc
import numpy as np
import tensorflow as tf

sys.path.append("../../models")
from deepchem.models.tensorgraph.layers import Layer, Feature, Label, L2LossLayer, AtomicConvolution, Transpose, Dense
from deepchem.models import TensorGraph

import numpy as np
import tensorflow as tf
import itertools
import time

class AtomicConvScore(Layer):
  def __init__(self, atom_types, **kwargs):
    self.atom_types = atom_types
    super(AtomicConvScore, self).__init__(**kwargs)

  def _create_tensor(self):
    frag1_z = self.in_layers[-3].out_tensor
    frag2_z = self.in_layers[-2].out_tensor
    complex_z = self.in_layers[-1].out_tensor

    B = frag1_z.get_shape()[0].value
    N_1 = frag1_z.get_shape()[1].value
    N_2 = frag2_z.get_shape()[1].value
    N = complex_z.get_shape()[1].value

    frag1_zeros = tf.zeros((B, N_1))
    frag2_zeros = tf.zeros((B, N_2))
    complex_zeros = tf.zeros((B, N))
    frag1_atomtype_energy = []
    frag2_atomtype_energy = []
    complex_atomtype_energy = []
    for index, atomtype in enumerate(self.atom_types):
      frag1_outputs = tf.squeeze(self.in_layers[index * 3].out_tensor)
      frag2_outputs = tf.squeeze(self.in_layers[index * 3 + 1].out_tensor)
      complex_outputs = tf.squeeze(self.in_layers[index * 3 + 2].out_tensor)

      cond = tf.equal(frag1_z, atomtype)
      frag1_atomtype_energy.append(tf.where(cond, frag1_outputs, frag1_zeros))
      cond = tf.equal(frag2_z, atomtype)
      frag2_atomtype_energy.append(tf.where(cond, frag2_outputs, frag2_zeros))
      cond = tf.equal(complex_z, atomtype)
      complex_atomtype_energy.append(tf.where(cond, complex_outputs, complex_zeros))

    frag1_outputs = tf.add_n(frag1_atomtype_energy)
    frag2_outputs = tf.add_n(frag2_atomtype_energy)
    complex_outputs = tf.add_n(complex_atomtype_energy)

    frag1_energy = tf.reduce_sum(frag1_outputs, 1)
    frag2_energy = tf.reduce_sum(frag2_outputs, 1)
    complex_energy = tf.reduce_sum(complex_outputs, 1)
    binding_energy = complex_energy - (frag1_energy + frag2_energy)
    self.out_tensor = tf.expand_dims(binding_energy, axis=1)
    return self.out_tensor

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "datasets")
train_dir = os.path.join(data_dir, "random_train")
test_dir = os.path.join(data_dir, "random_test")
model_dir = os.path.join(base_dir, "random_model")

frag1_num_atoms = 70
frag2_num_atoms = 634
complex_num_atoms = 701
max_num_neighbors = 12
neighbor_cutoff = 12.0
batch_size=100

train_dataset = dc.data.DiskDataset(train_dir)
test_dataset = dc.data.DiskDataset(test_dir)
pdbbind_tasks = ["-logKd/Ki"]
transformers = []
#transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
#for transformer in transformers:
#  train_dataset = transformer.transform(train_dataset)
#  test_dataset = transformer.transform(test_dataset)

y_train = train_dataset.y
y_train *= -1*2.479/4.184
train_dataset = dc.data.DiskDataset.from_numpy(train_dataset.X, y_train, train_dataset.w, train_dataset.ids, tasks=pdbbind_tasks)

y_test = test_dataset.y
y_test *= -1*2.479/4.184
test_dataset = dc.data.DiskDataset.from_numpy(test_dataset.X, y_test, test_dataset.w, test_dataset.ids, tasks=pdbbind_tasks)

at = [6,  7.,  8.,  9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]
radial = [[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0], [0.0, 4.0, 8.0], [0.4]]
rp = [x for x in itertools.product(*radial)]
layer_sizes = [32, 32, 16]
dropouts = [0., 0., 0.]
penalty = 0.

frag1_X = Feature(shape=(batch_size, frag1_num_atoms, 3))
frag1_nbrs = Feature(shape=(batch_size, frag1_num_atoms, max_num_neighbors))
frag1_nbrs_z = Feature(shape=(batch_size, frag1_num_atoms, max_num_neighbors))
frag1_z = Feature(shape=(batch_size, frag1_num_atoms))

frag2_X = Feature(shape=(batch_size, frag2_num_atoms, 3))
frag2_nbrs = Feature(shape=(batch_size, frag2_num_atoms, max_num_neighbors))
frag2_nbrs_z = Feature(shape=(batch_size, frag2_num_atoms, max_num_neighbors))
frag2_z = Feature(shape=(batch_size, frag2_num_atoms))

complex_X = Feature(shape=(batch_size, complex_num_atoms, 3))
complex_nbrs = Feature(shape=(batch_size, complex_num_atoms, max_num_neighbors))
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

score_in_layers = []
for atom_type in at:
  at_frag1_conv = frag1_conv
  at_frag2_conv = frag2_conv
  at_complex_conv = complex_conv
  for layer_size in layer_sizes:
    at_frag1_conv = Dense(
      out_channels=layer_size,
      activation_fn=tf.nn.relu,
      time_series=True,
      in_layers=[at_frag1_conv])
    at_frag2_conv = at_frag1_conv.shared(in_layers=[at_frag2_conv])
    at_complex_conv = at_frag1_conv.shared(in_layers=[at_complex_conv])
  at_frag1_conv = Dense(
    out_channels=1,
    activation_fn=None,
    time_series=True,
    in_layers=[at_frag1_conv])
  at_frag2_conv = at_frag1_conv.shared(in_layers=[at_frag2_conv])
  at_complex_conv = at_frag1_conv.shared(in_layers=[at_complex_conv])
  score_in_layers.append(at_frag1_conv)
  score_in_layers.append(at_frag2_conv)
  score_in_layers.append(at_complex_conv)

score_in_layers.extend([frag1_z, frag2_z, complex_z])
score = AtomicConvScore(atom_types=at, in_layers=score_in_layers)

label = Label(shape=(None, 1))
loss = L2LossLayer(in_layers=[score, label])


def feed_dict_generator(dataset, batch_size, epochs=1):
  for epoch in range(epochs):
    for ind, (F_b, y_b, w_b, ids_b
              ) in enumerate(dataset.iterbatches(batch_size, deterministic=True, pad_batches=True)):
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
      orig_dict[frag1_z] = frag1_Z_b

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
      orig_dict[frag2_z] = frag2_Z_b

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
      orig_dict[complex_z] = complex_Z_b
      orig_dict[label] = np.reshape(y_b, newshape=(batch_size, 1))
      yield orig_dict


tg = TensorGraph(
  batch_size=batch_size,
  mode=str("regression"),
  model_dir=str("/tmp/atom_conv"))
tg.add_output(score)
tg.set_loss(loss)

print("Fitting")
metric = [
  dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
  dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]
tg.fit_generator(feed_dict_generator(train_dataset, batch_size, epochs=10))

train_evaluator = dc.utils.evaluate.GeneratorEvaluator(
  tg, feed_dict_generator(train_dataset, batch_size), transformers, [label])
train_scores = train_evaluator.compute_model_performance(metric)
print("Train scores")
print(train_scores)
test_evaluator = dc.utils.evaluate.GeneratorEvaluator(
  tg, feed_dict_generator(test_dataset, batch_size), transformers, [label])
test_scores = test_evaluator.compute_model_performance(metric)
print("Test scores")
print(test_scores)
