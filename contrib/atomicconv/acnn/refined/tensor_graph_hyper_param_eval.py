from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import deepchem as dc
import json
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

batch_size = 24
splits = ["random", "stratified", "index", "scaffold"]


def params():
  d2 = {
      "name":
      "hyper5",
      "radial": [[
          1.5, 2.5, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
          10.5
      ], [0.0, 2.0, 4.0], [0.05]],
      "layer_sizes": [32, 16, 8],
      "learning_rate":
      0.001,
      "epochs":
      10,
      "frag1_num_atoms":
      71,
      "frag2_num_atoms":
      912,
      "complex_num_atoms":
      982,
  }
  yield d2
  d2 = {
      "name":
      "original",
      "radial": [[
          1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
          8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
      ], [0.0, 4.0, 8.0], [0.4]],
      "layer_sizes": [32, 32, 16],
      "learning_rate":
      0.001,
      "epochs":
      10,
      "frag1_num_atoms":
      71,
      "frag2_num_atoms":
      912,
      "complex_num_atoms":
      982,
  }
  yield d2


metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]
for split in splits:
  data_dir = os.path.join(base_dir, "datasets")
  train_dir = os.path.join(data_dir, "%s_train" % split)
  valid_dir = os.path.join(data_dir, "%s_valid" % split)
  test_dir = os.path.join(data_dir, "%s_test" % split)

  train_dataset = dc.data.DiskDataset(train_dir)
  valid_dataset = dc.data.DiskDataset(train_dir)
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
  for param in params():
    num_epochs = param['epochs']
    del param['epochs']
    name = param['name']
    del param['name']

    tg, feed_dict_generator, label = atomic_conv_model(**param)
    for i in range(2):
      tg.fit_generator(
          feed_dict_generator(train_dataset, batch_size, epochs=num_epochs))

      valid_evaluator = dc.utils.evaluate.GeneratorEvaluator(
          tg,
          feed_dict_generator(valid_dataset, batch_size), transformers, [label])
      valid_scores = valid_evaluator.compute_model_performance(metric)
      valid_scores = {"%s_valid" % k: v for k, v in valid_scores.items()}
      param.update(valid_scores)
      # test_evaluator = dc.utils.evaluate.GeneratorEvaluator(
      #   tg, feed_dict_generator(test_dataset, batch_size), transformers, [label])
      # test_scores = test_evaluator.compute_model_performance(metric)
      # test_scores = {"%s_test" % k: v for k, v in test_scores.items()}
      # param.update(test_scores)
      param['epochs'] = num_epochs * (i + 1)
      param['split'] = split
      param['name'] = name
      print("Results")
      print(param)
      with open('hyper_evaluation.txt', 'a') as fout:
        fout.write(json.dumps(param))
        fout.write("\n")
