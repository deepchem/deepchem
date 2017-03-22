# DeepCrystal Technologies 2017 - Patrick Hop
# MIT License - have fun!!

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
np.random.seed(123)
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

BATCH_SIZE = 128
MAX_EPOCH = 40
LR = 1e-3
LMBDA = 1e-4


def retrieve_datasets():
  os.system(
      'wget %s' %
      'https://s3-us-west-1.amazonaws.com/deep-crystal-california/az_logd.csv')
  os.system(
      'wget %s' %
      'https://s3-us-west-1.amazonaws.com/deep-crystal-california/az_hppb.csv')
  os.system(
      'wget %s' %
      'https://s3-us-west-1.amazonaws.com/deep-crystal-california/az_clearance.csv'
  )


def load_dataset(dataset_file, featurizer='ECFP', split='index'):
  tasks = ['exp']

  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=4096)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()

  loader = dc.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  transformers = [
      dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)
  ]
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return tasks, (train, valid, test), transformers


def experiment(dataset_file, method='GraphConv', split='scaffold'):
  featurizer = 'ECFP'
  if method == 'GraphConv':
    featurizer = 'GraphConv'
  tasks, datasets, transformers = load_dataset(
      dataset_file, featurizer=featurizer, split=split)
  train, val, test = datasets

  model = None
  if method == 'GraphConv':
    n_feat = 75
    graph_model = dc.nn.SequentialGraph(n_feat)
    graph_model.add(dc.nn.GraphConv(128, n_feat, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphPool())

    graph_model.add(dc.nn.GraphConv(128, 128, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphPool())

    graph_model.add(dc.nn.Dense(256, 128, activation='relu'))
    graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(dc.nn.GraphGather(BATCH_SIZE, activation="tanh"))

    model = dc.models.MultitaskGraphRegressor(
        graph_model,
        len(tasks),
        n_feat,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)
  elif method == 'PDNN':
    model = dc.models.TensorflowMultiTaskRegressor(
        len(tasks),
        train.get_data_shape()[0],
        layer_sizes=[384, 196],
        dropouts=[.25, .25],
        weight_init_stddevs=[.02, .02],
        bias_init_consts=[.1, .1],
        learning_rate=LR,
        penalty=LMBDA,
        penalty_type="l2",
        optimizer="adam",
        batch_size=BATCH_SIZE,
        seed=123,
        verbosity="high")
  elif method == 'RF':

    def model_builder_rf(model_dir):
      sklearn_model = RandomForestRegressor(n_estimators=100)
      return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(tasks, model_builder_rf)
  elif method == 'SVR':

    def model_builder_svr(model_dir):
      sklearn_model = svm.SVR(kernel='linear')
      return dc.models.SklearnModel(sklearn_model, model_dir)

    model = dc.models.SingletaskToMultitask(tasks, model_builder_svr)

  return model, train, val, test, transformers


#======================================================================
# Run Benchmarks {GC-DNN, P-DNN, SVR, RF}

retrieve_datasets()

MODEL = 'GraphConv'
SPLIT = 'scaffold'
DATASET = 'az_hppb.csv'

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
model, train, val, test, transformers = experiment(
    DATASET, method=MODEL, split=SPLIT)
if MODEL == 'GraphConv':
  best_val_score = 0.0
  train_score = 0.0
  for l in xrange(0, MAX_EPOCH):
    model.fit(train, nb_epoch=1)
    latest_train_score = model.evaluate(train, [metric],
                                        transformers)['mean-pearson_r2_score']
    latest_val_score = model.evaluate(val, [metric],
                                      transformers)['mean-pearson_r2_score']
    if latest_val_score > best_val_score:
      best_val_score = latest_val_score
      train_score = latest_train_score
  print((MODEL, SPLIT, DATASET, train_score, best_val_score))
elif MODEL == 'PDNN':
  model.fit(train, nb_epoch=25)
  train_score = model.evaluate(train, [metric],
                               transformers)['mean-pearson_r2_score']
  val_score = model.evaluate(val, [metric],
                             transformers)['mean-pearson_r2_score']
  print((MODEL, SPLIT, DATASET, train_score, val_score))
  # we cant re-open the closed session...
else:
  model.fit(train)
  train_score = model.evaluate(train, [metric],
                               transformers)['mean-pearson_r2_score']
  val_score = model.evaluate(val, [metric],
                             transformers)['mean-pearson_r2_score']
  print((MODEL, SPLIT, DATASET, train_score, val_score))
