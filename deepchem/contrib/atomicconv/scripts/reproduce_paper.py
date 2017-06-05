__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import argparse

import numpy as np
import tensorflow as tf

import deepchem as dc
from deepchem.utils.evaluate import Evaluator

from deepchem.contrib.atomicconv.models.publication import AtomicConvModel
from deepchem.contrib.atomicconv.acnn.atomicnet_ops import create_symmetry_parameters

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_dir', required=True)
parser.add_argument('-o', '--outdir', required=True)
parser.add_argument('-set', '--which_set', required=True)
parser.add_argument('-split', '--which_split', required=True)
args = parser.parse_args()

# Setup
seed = 125
np.random.seed(seed)
tf.set_random_seed(seed)
pdbbind_tasks = ["-logKd/Ki"]

# TODO: Create args so that this script can be general across datasets/splits
which_set = args.which_set
which_split = args.which_split

# Setup
data_dir = os.path.join(args.dataset_dir)
train_dir = os.path.join(data_dir, which_set, "{}_train".format(which_split))
test_dir = os.path.join(data_dir, which_set, "{}_test".format(which_split))
model_dir = os.path.join(args.outdir, "{}_{}_model".format(which_set, which_split))

# Hyperparameters
if which_set == "core":
    frag1_num_atoms = 140
    frag2_num_atoms = 821
    complex_num_atoms = 908
    max_num_neighbors = 12
    neighbor_cutoff = 12.0
    at = [1., 6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]
    radial = [[12.0], [0.0, 4.0, 8.0], [4.0]]
    layer_sizes = [32, 32, 16, 1]
    dropouts = [0., 0., 0., 0.]
    penalty_type = "l2"
    penalty = 0.
    num_training_epochs = 100
    weight_init_stddevs = [1 / np.sqrt(layer_sizes[0]),
                           1 / np.sqrt(layer_sizes[1]),
                           1 / np.sqrt(layer_sizes[2]),
                           0.01]
    bias_init_consts = [0., 0., 0., 0.]
    learning_rate=0.002

elif which_set == "refined":
    frag1_num_atoms = 153
    frag2_num_atoms = 1119
    complex_num_atoms = 1254
    max_num_neighbors = 12
    neighbor_cutoff = 12.0
    at = [1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27.,
         28., 29., 30., 34., 35., 38., 48., 53., 55., 80.]
    radial = [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]]
    layer_sizes = [128, 128, 64, 1]
    dropouts = [0.4, 0.4, 0.]
    penalty_type = "l2"
    penalty = 0.
    num_training_epochs = 100
    weight_init_stddevs = [0.125, 0.125, 0.177, 0.01]
    bias_init_consts = [0., 0., 0., 0.]
    learning_rate=0.002


train_dataset = dc.data.DiskDataset(train_dir)
test_dataset = dc.data.DiskDataset(test_dir)

transformers = [
    dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
]
for transformer in transformers:
  train_dataset = transformer.transform(train_dataset)
  test_dataset = transformer.transform(test_dataset)

# Setup graph
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    model = AtomicConvModel(len(pdbbind_tasks),
                            create_symmetry_parameters(radial),
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
                            dropouts=dropouts, # TODO: FIX DROPOUT
                            learning_rate=learning_rate,
                            momentum=0.8,
                            optimizer="adam",
                            batch_size=24,
                            conv_layers=1,
                            boxsize=None,
                            verbose=True,
                            seed=seed,
                            sess=sess)

    model.fit(train_dataset, nb_epoch=num_training_epochs, sess=sess)

    metric = [
        dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
        dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
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


