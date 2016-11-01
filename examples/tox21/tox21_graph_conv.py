"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
from keras import backend as K
from tox21_datasets import load_tox21_convmol

# Only for debug!
np.random.seed(123)

g = tf.Graph()
sess = tf.Session(graph=g)
K.set_session(sess)

with g.as_default():

  # Set some global variables up top
  verbosity = "high"

  #Make directories to store the raw and featurized datasets.
  base_dir = "/tmp/tox21_tf"
  # This is for good debug (to make sure nasty state isn't being passed around)
  if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
  os.makedirs(base_dir)
  data_dir = os.path.join(base_dir, "dataset")
  model_dir = os.path.join(base_dir, "model")

  # Load Tox21 dataset
  n_features = 1024
  tox21_tasks, tox21_datasets, transformers = load_tox21_convmol(data_dir)
  # Do train/valid split.
  train_dataset, valid_dataset = tox21_datasets

  # Fit models
  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean,
                                            verbosity=verbosity,
                                            mode="classification")

  # Do setup required for tf/keras models
  # Number of features on conv-mols
  n_feat = 71
  # Batch size of models
  batch_size = 50
  graph_model = dc.models.SequentialGraphModel(n_feat)
  graph_model.add(dc.nn.GraphConv(64, activation='relu'))
  graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  graph_model.add(dc.nn.GraphPool())
  graph_model.add(dc.nn.GraphConv(64, activation='relu'))
  graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  graph_model.add(dc.nn.GraphPool())
  # Gather Projection
  graph_model.add(dc.nn.Dense(128, activation='relu'))
  graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
  graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))
  # Dense post-processing layer

  with tf.Session() as sess:
    model = dc.models.MultitaskGraphClassifier(
      sess, graph_model, len(tox21_tasks), model_dir, batch_size=batch_size,
      learning_rate=1e-3, learning_rate_decay_time=1000,
      optimizer_type="adam", beta1=.9, beta2=.999, verbosity="high")

    # Fit trained model
    model.fit(train_dataset, nb_epoch=10)

    train_scores = model.evaluate(train_dataset, [classification_metric])
    print("Train scores")
    print(train_scores)

    valid_scores = model.evaluate(valid_dataset, [classification_metric])
    print("Validation scores")
    print(valid_scores)
