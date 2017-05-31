"""
Script that trains graph-conv models on SuperToxic dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import csv
import time 
time_start = time.time()

import numpy as np
import tensorflow as tf
import deepchem as dc

from keras import backend as K

from supertoxic_datasets import load_supertoxic

# Only for debug!
np.random.seed(123)

g = tf.Graph()
sess = tf.Session(graph=g)
K.set_session(sess)


with g.as_default():

  # Load SuperToxic dataset
  tf.set_random_seed(123)
  supertoxic_tasks, supertoxic_datasets, transformers = load_supertoxic(featurizer='GraphConv', split='random')
  train_dataset, valid_dataset, test_dataset = supertoxic_datasets

  print("SuperToxic Tasks: {}".format(supertoxic_tasks))

  # Fit models
  metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

  print("Constructing graph")
  # Number of features on conv-mols
  n_feat = 75
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

    print("Building model")
    model = dc.models.MultitaskGraphRegressor(
      sess, graph_model, len(supertoxic_tasks), batch_size=batch_size,
      learning_rate=1e-3, learning_rate_decay_time=1000,
      optimizer_type="adam", beta1=.9, beta2=.999)

    # Fit trained model
    print("Fitting model")
    nb_epoch = 20
    model.fit(train_dataset, nb_epoch=nb_epoch)

    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [metric], transformers)
    valid_scores = model.evaluate(valid_dataset, [metric], transformers)

    print("Train scores")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)
    pred = model.predict(train_dataset, transformers)
    pred2 = model.predict(valid_dataset, transformers)
    pred3 = model.predict(test_dataset, transformers)


    with open('./results.csv','a') as f:
      writer = csv.writer(f)
      output_line = [os.path.basename(__file__), 'nb_epoch', nb_epoch,
                     'train', train_scores['mean-pearson_r2_score'],
                     'valid', valid_scores['mean-pearson_r2_score'], time.time()-time_start]
      writer.writerow(output_line)
      print("Saving results")
      print("file: results.csv")
      print("results:\t{}"
            .format(map(str, output_line)))
