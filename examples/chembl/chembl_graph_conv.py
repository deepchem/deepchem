"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
from keras import backend as K
from chembl_datasets import load_chembl

# Only for debug!
np.random.seed(123)

g = tf.Graph()
sess = tf.Session(graph=g)
K.set_session(sess)

with g.as_default():
  tf.set_random_seed(123)
  chembl_tasks, datasets, transformers = load_chembl(
      featurizer='GraphConv', split='index')
  train_dataset, valid_dataset, test_dataset = datasets

  # Fit models
  metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

  # Do setup required for tf/keras models
  # Number of features on conv-mols
  n_feat = 71
  # Batch size of models
  batch_size = 128
  graph_model = dc.nn.SequentialGraph(n_feat)
  graph_model.add(dc.nn.GraphConv(128, activation='relu'))
  graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  graph_model.add(dc.nn.GraphPool())
  graph_model.add(dc.nn.GraphConv(128, activation='relu'))
  graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  graph_model.add(dc.nn.GraphPool())
  # Gather Projection
  graph_model.add(dc.nn.Dense(256, activation='relu'))
  graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))
  # Dense post-processing layer

  with tf.Session() as sess:
    model = dc.models.MultitaskGraphRegressor(
      sess, graph_model, len(chembl_tasks), batch_size=batch_size,
      learning_rate=1e-3, learning_rate_decay_time=1000,
      optimizer_type="adam", beta1=.9, beta2=.999)

    # Fit trained model
    model.fit(train_dataset, nb_epoch=10)

    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [metric], transformers)
    valid_scores = model.evaluate(valid_dataset, [metric], transformers)
    # Only use for final evaluation
    test_scores = model.evaluate(test_dataset, [metric], transformers)

    print("Train scores")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)

    print("Test scores")
    print(test_scores)
