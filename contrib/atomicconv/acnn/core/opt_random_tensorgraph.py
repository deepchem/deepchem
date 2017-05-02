from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import deepchem as dc
import numpy as np
import tensorflow as tf
from deepchem.models.tensorgraph.models.atomic_conv import atomic_conv_model

sys.path.append("../../models")
from deepchem.models.tensorgraph.layers import Layer, Feature, Label, L2Loss, AtomicConvolution, Transpose, Dense
from deepchem.models import TensorGraph

import numpy as np
import tensorflow as tf
import itertools
import time

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "datasets")
train_dir = os.path.join(data_dir, "random_train")
test_dir = os.path.join(data_dir, "random_test")
model_dir = os.path.join(base_dir, "random_model")

train_dataset = dc.data.DiskDataset(train_dir)
test_dataset = dc.data.DiskDataset(test_dir)
pdbbind_tasks = ["-logKd/Ki"]
transformers = []

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

batch_size = 24
tg, feed_dict_generator, label = atomic_conv_model()

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
