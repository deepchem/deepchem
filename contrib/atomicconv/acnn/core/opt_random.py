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

train_dataset = dc.data.DiskDataset(train_dir)
test_dataset = dc.data.DiskDataset(test_dir)
pdbbind_tasks = ["-logKd/Ki"]
transformers = []
#transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
#for transformer in transformers:
#  train_dataset = transformer.transform(train_dataset)
#  test_dataset = transformer.transform(test_dataset)

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

at = [6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]
radial = [[
    1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
    9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
], [0.0, 4.0, 8.0], [0.4]]
#radial = [[12.0], [0.0, 4.0, 8.0], [0.4]]
rp = create_symmetry_parameters(radial)
layer_sizes = [32, 32, 16]
weight_init_stddevs = [
    1 / np.sqrt(layer_sizes[0]), 1 / np.sqrt(layer_sizes[1]),
    1 / np.sqrt(layer_sizes[2])
]
dropouts = [0.3, 0.3, 0.05]
penalty_type = "l2"
penalty = 0.
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
    bias_init_consts=[0., 0., 0.],
    penalty=penalty,
    penalty_type=penalty_type,
    dropouts=dropouts,
    learning_rate=0.002,
    momentum=0.8,
    optimizer="adam",
    batch_size=24,
    conv_layers=1,
    boxsize=None,
    verbose=True,
    seed=seed)
model.fit(train_dataset, nb_epoch=10)
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]
train_evaluator = dc.utils.evaluate.Evaluator(model, train_dataset,
                                              transformers)
train_scores = train_evaluator.compute_model_performance(
    metric,
    csv_out="train_predict_ac_random.csv",
    stats_out="train_stats_ac_random.csv")
print("Train scores")
print(train_scores)
test_evaluator = dc.utils.evaluate.Evaluator(model, test_dataset, transformers)
test_scores = test_evaluator.compute_model_performance(
    metric,
    csv_out="test_predict_ac_random.csv",
    stats_out="test_stats_ac_random.csv")
print("Test scores")
print(test_scores)
