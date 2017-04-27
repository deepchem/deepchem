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
from atomicnet_ops import create_symmetry_parameters
from atomicnet import TensorflowFragmentRegressor

# Set random seeds
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Setup directories
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "datasets")
train_dir = os.path.join(data_dir, "random_train")
test_dir = os.path.join(data_dir, "random_test")
model_dir = os.path.join(base_dir, "random_model")

# Model constants
frag1_num_atoms = 153
frag2_num_atoms = 1119
complex_num_atoms = 1254
max_num_neighbors = 12
neighbor_cutoff = 12.0

# Load and transform datasets
pdbbind_tasks = ["-logKd/Ki"]
train_dataset = dc.data.DiskDataset(train_dir)
test_dataset = dc.data.DiskDataset(test_dir)

transformers = []
# convert -logKi to dG = +RTlogKi [kJ/mol]
y_train = train_dataset.y
y_train *= -1 * 2.479 / 4.184
train_dataset = dc.data.DiskDataset.from_numpy(
    train_dataset.X,
    y_train,
    train_dataset.w,
    train_dataset.ids,
    tasks=pdbbind_tasks)
y_test = test_dataset.y
y_test *= -1 * 2.479 / 4.184
test_dataset = dc.data.DiskDataset.from_numpy(
    test_dataset.X,
    y_test,
    test_dataset.w,
    test_dataset.ids,
    tasks=pdbbind_tasks)

# Atomic convolution variables
# at = atomic numbers (atom types)
# radial basis function parameters [cutoff, mean, width]
at = [
    1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
    29., 30., 34., 35., 38., 48., 53., 55., 80.
]
radial = [[1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
          [0.0], [0.4]]
rp = create_symmetry_parameters(radial)

# Model hyperparameters
layer_sizes = [32, 32, 16]
weight_init_stddevs = [1 / np.sqrt(x) for x in layer_sizes]
bias_init_consts = [0. for x in layer_sizes]
penalty_type = "l2"
penalty = 0.
dropouts = [0.25, 0.25, 0.]
learning_rate = 0.001
momentum = 0.8
batch_size = 24

# Initialize model
model = TensorflowFragmentRegressor(
    len(pdbbind_tasks),
    rp,
    at,
    frag1_num_atoms,
    frag2_num_atoms,
    complex_num_atoms,
    max_num_neighbors,
    logdir=model_dir,
    layer_sizes=layer_sizes,
    weight_init_stddevs=weight_init_stddevs,
    bias_init_consts=bias_init_consts,
    penalty=penalty,
    penalty_type=penalty_type,
    dropouts=dropouts,
    learning_rate=learning_rate,
    momentum=momentum,
    optimizer="adam",
    batch_size=batch_size,
    conv_layers=1,
    boxsize=None,
    verbose=True,
    seed=seed)

# Fit model
model.fit(train_dataset, nb_epoch=10)

# Evaluate model
metric = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression"),
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")
]
train_evaluator = dc.utils.evaluate.Evaluator(model, train_dataset,
                                              transformers)
train_scores = train_evaluator.compute_model_performance(metric)
print("Train scores")
print(train_scores)
test_evaluator = dc.utils.evaluate.Evaluator(model, test_dataset, transformers)
test_scores = test_evaluator.compute_model_performance(metric)
print("Test scores")
print(test_scores)
