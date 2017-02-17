#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:53:27 2016

@author: Michael Wu

Benchmark test:

Giving classification performances of: 
    Random forest(rf), MultitaskDNN(tf), 
    RobustMultitaskDNN(tf_robust),
    Logistic regression(logreg), IRV(irv)
    Graph convolution(graphconv)                 
on datasets: muv, pcba, tox21, sider, toxcast, clintox

Giving regression performances of:
    MultitaskDNN(tf_regression),
    Graph convolution regression(graphconvreg)
on datasets: delaney(ESOL), nci, kaggle, pdbbind, qm7, 
             chembl, sampl(FreeSolv)

time estimation listed in README file

Total time of running a benchmark test(for one splitting function): 20h
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import os
import numpy as np
import shutil
import time
import deepchem as dc
import tensorflow as tf
import argparse
from keras import backend as K
import csv
from sklearn.ensemble import RandomForestClassifier

from muv.muv_datasets import load_muv
from nci.nci_datasets import load_nci
from pcba.pcba_datasets import load_pcba
from tox21.tox21_datasets import load_tox21
from toxcast.toxcast_datasets import load_toxcast
from sider.sider_datasets import load_sider
from kaggle.kaggle_datasets import load_kaggle
from delaney.delaney_datasets import load_delaney
from pdbbind.pdbbind_datasets import load_pdbbind_grid
from chembl.chembl_datasets import load_chembl
from qm7.qm7_datasets import load_qm7
from sampl.sampl_datasets import load_sampl
from clintox.clintox_datasets import load_clintox


def benchmark_loading_datasets(hyper_parameters,
                               dataset='tox21',
                               model='tf',
                               split=None,
                               reload=True,
                               out_path='.',
                               test=False):
  """
  Loading dataset for benchmark test
  
  Parameters
  ----------
  hyper_parameters: dict
      hyper parameters including layer size, dropout, learning rate, etc.
  dataset: string, optional (default='tox21')
      choice of which dataset to use, should be: tox21, muv, sider, 
      toxcast, pcba, delaney, kaggle, nci, clintox, pdbbind, chembl,
      qm7, sampl
  model: string,  optional (default='tf')
      choice of which model to use, should be: rf, tf, tf_robust, logreg,
      irv, graphconv, tf_regression, graphconvreg
  split: string,  optional (default=None)
      choice of splitter function, None = using the default splitter
  out_path: string, optional(default='.')
      path of result file
  """

  if dataset in ['muv', 'pcba', 'tox21', 'sider', 'toxcast', 'clintox']:
    mode = 'classification'
  elif dataset in [
      'kaggle', 'delaney', 'nci', 'pdbbind', 'chembl', 'qm7', 'sampl'
  ]:
    mode = 'regression'
  else:
    raise ValueError('Dataset not supported')

  # Assigning featurizer
  if model in ['graphconv', 'graphconvreg']:
    featurizer = 'GraphConv'
    n_features = 75
  elif model in ['tf', 'tf_robust', 'logreg', 'rf', 'irv', 'tf_regression']:
    featurizer = 'ECFP'
    n_features = 1024
  else:
    raise ValueError('Model not supported')

  # Some exceptions in datasets
  if dataset in ['kaggle']:
    featurizer = None  # kaggle dataset use its own features
    if split in ['random', 'scaffold']:
      return
    else:
      split = None  # kaggle dataset is already splitted
    if not model in ['tf_regression']:
      return

  if dataset in ['pdbbind']:
    featurizer = 'grid'  # pdbbind use grid featurizer
    if split in ['scaffold', 'index']:
      return  # skip the scaffold and index splitting of pdbbind
    if not model in ['tf_regression']:
      return

  if split in ['year']:
    if not dataset in ['chembl']:
      return
  elif not split in [None, 'index', 'random', 'scaffold', 'butina']:
    raise ValueError('Splitter function not supported')

  loading_functions = {
      'tox21': load_tox21,
      'muv': load_muv,
      'pcba': load_pcba,
      'nci': load_nci,
      'sider': load_sider,
      'toxcast': load_toxcast,
      'kaggle': load_kaggle,
      'delaney': load_delaney,
      'pdbbind': load_pdbbind_grid,
      'chembl': load_chembl,
      'qm7': load_qm7,
      'sampl': load_sampl,
      'clintox': load_clintox
  }

  print('-------------------------------------')
  print('Benchmark %s on dataset: %s' % (model, dataset))
  print('-------------------------------------')
  time_start = time.time()
  # loading datasets
  if split is not None:
    print('Splitting function: %s' % split)
    tasks, all_dataset, transformers = loading_functions[dataset](
        featurizer=featurizer, split=split)
  else:
    tasks, all_dataset, transformers = loading_functions[dataset](
        featurizer=featurizer)

  train_dataset, valid_dataset, test_dataset = all_dataset
  time_finish_loading = time.time()
  # time_finish_loading-time_start is the time(s) used for dataset loading
  if dataset in ['kaggle', 'pdbbind', 'qm7']:
    n_features = train_dataset.get_data_shape()[0]
    # dataset has customized features

  # running model
  for count, hp in enumerate(hyper_parameters[model]):
    time_start_fitting = time.time()
    if mode == 'classification':
      metric = 'auc'
      train_score, valid_score, test_score = benchmark_classification(
          train_dataset,
          valid_dataset,
          test_dataset,
          tasks,
          transformers,
          hp,
          n_features,
          metric=metric,
          model=model,
          test=test)
    elif mode == 'regression':
      metric = 'r2'
      train_score, valid_score, test_score = benchmark_regression(
          train_dataset,
          valid_dataset,
          test_dataset,
          tasks,
          transformers,
          hp,
          n_features,
          metric=metric,
          model=model,
          test=test)
    time_finish_fitting = time.time()

    with open(os.path.join(out_path, 'results.csv'), 'a') as f:
      writer = csv.writer(f)
      if mode == 'classification':
        for i in train_score:
          output_line = [
              count, dataset, str(split), mode, 'train', i,
              train_score[i]['mean-roc_auc_score'], 'valid', i,
              valid_score[i]['mean-roc_auc_score']
          ]
          if test:
            output_line.extend(['test', i, test_score[i]['mean-roc_auc_score']])
          output_line.extend(
              ['time_for_running', time_finish_fitting - time_start_fitting])
          writer.writerow(output_line)
      else:
        for i in train_score:
          if metric == 'r2':
            output_line = [
                count, dataset, str(split), mode, 'train', i,
                train_score[i]['mean-pearson_r2_score'], 'valid', i,
                valid_score[i]['mean-pearson_r2_score']
            ]
            if test:
              output_line.extend(
                  ['test', i, test_score[i]['mean-pearson_r2_score']])
            output_line.extend(
                ['time_for_running', time_finish_fitting - time_start_fitting])
          elif metric == 'mae':
            output_line = [
                count, dataset, str(split), mode, 'train', i,
                train_score[i]['mean-mean_absolute_error'], 'valid', i,
                valid_score[i]['mean-mean_absolute_error']
            ]
            if test:
              output_line.extend(
                  ['test', i, test_score[i]['mean-mean_absolute_error']])
            output_line.extend(
                ['time_for_running', time_finish_fitting - time_start_fitting])

          writer.writerow(output_line)


def benchmark_classification(train_dataset,
                             valid_dataset,
                             test_dataset,
                             tasks,
                             transformers,
                             hyper_parameters,
                             n_features,
                             metric='auc',
                             model='tf',
                             seed=123,
                             test=False):
  """
  Calculate performance of different models on the specific dataset & tasks
  
  Parameters
  ----------
  train_dataset: dataset struct
      loaded dataset using load_* or splitter function
  valid_dataset: dataset struct
      loaded dataset using load_* or splitter function
  tasks: list of string
      list of targets(tasks, datasets)
  transformers: BalancingTransformer struct
      loaded properties of dataset from load_* function
  hyper_parameters: dict
      hyper parameters including layer size, dropout, learning rate, etc.
  n_features: integer
      number of features, or length of binary fingerprints
  model: string,  optional (default='tf')
      choice of which model to use, should be: rf, tf, tf_robust, logreg,
      irv, graphconv
  test: boolean
      whether to calculate test_set performance
  

  Returns
  -------
  train_scores : dict
	predicting results(AUC) on training set
  valid_scores : dict
	predicting results(AUC) on valid set
  test_scores : dict
	predicting results(AUC) on test set
 

  """
  train_scores = {}
  valid_scores = {}
  test_scores = {}

  # Initialize metrics
  if metric == 'auc':
    classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

  assert model in ['rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv']

  if model == 'tf':
    # Loading hyper parameters
    layer_sizes = hyper_parameters['layer_sizes']
    weight_init_stddevs = hyper_parameters['weight_init_stddevs']
    bias_init_consts = hyper_parameters['bias_init_consts']
    dropouts = hyper_parameters['dropouts']
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']

    # Building tensorflow MultiTaskDNN model
    model_tf = dc.models.TensorflowMultiTaskClassifier(
        len(tasks),
        n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

    print('-------------------------------------')
    print('Start fitting by multitask DNN')
    model_tf.fit(train_dataset, nb_epoch=nb_epoch)

    # Evaluating tensorflow MultiTaskDNN model
    train_scores['tf'] = model_tf.evaluate(
        train_dataset, [classification_metric], transformers)

    valid_scores['tf'] = model_tf.evaluate(
        valid_dataset, [classification_metric], transformers)

    if test:
      test_scores['tf'] = model_tf.evaluate(
          test_dataset, [classification_metric], transformers)

  if model == 'tf_robust':
    # Loading hyper parameters
    layer_sizes = hyper_parameters['layer_sizes']
    weight_init_stddevs = hyper_parameters['weight_init_stddevs']
    bias_init_consts = hyper_parameters['bias_init_consts']
    dropouts = hyper_parameters['dropouts']

    bypass_layer_sizes = hyper_parameters['bypass_layer_sizes']
    bypass_weight_init_stddevs = hyper_parameters['bypass_weight_init_stddevs']
    bypass_bias_init_consts = hyper_parameters['bypass_bias_init_consts']
    bypass_dropouts = hyper_parameters['bypass_dropouts']

    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']

    # Building tensorflow robust MultiTaskDNN model
    model_tf_robust = dc.models.RobustMultitaskClassifier(
        len(tasks),
        n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        bypass_layer_sizes=bypass_layer_sizes,
        bypass_weight_init_stddevs=bypass_weight_init_stddevs,
        bypass_bias_init_consts=bypass_bias_init_consts,
        bypass_dropouts=bypass_dropouts,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

    print('--------------------------------------------')
    print('Start fitting by robust multitask DNN')
    model_tf_robust.fit(train_dataset, nb_epoch=nb_epoch)

    # Evaluating tensorflow robust MultiTaskDNN model
    train_scores['tf_robust'] = model_tf_robust.evaluate(
        train_dataset, [classification_metric], transformers)

    valid_scores['tf_robust'] = model_tf_robust.evaluate(
        valid_dataset, [classification_metric], transformers)

    if test:
      test_scores['tf_robust'] = model_tf_robust.evaluate(
          test_dataset, [classification_metric], transformers)

  if model == 'logreg':
    # Loading hyper parameters
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']

    # Building tensorflow logistic regression model
    model_logreg = dc.models.TensorflowLogisticRegression(
        len(tasks),
        n_features,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

    print('-------------------------------------')
    print('Start fitting by logistic regression')
    model_logreg.fit(train_dataset, nb_epoch=nb_epoch)

    # Evaluating tensorflow logistic regression model
    train_scores['logreg'] = model_logreg.evaluate(
        train_dataset, [classification_metric], transformers)

    valid_scores['logreg'] = model_logreg.evaluate(
        valid_dataset, [classification_metric], transformers)

    if test:
      test_scores['logreg'] = model_logreg.evaluate(
          test_dataset, [classification_metric], transformers)

  if model == 'irv':
    # Loading hyper parameters
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_K = hyper_parameters['n_K']

    # Transform fingerprints to IRV features
    transformer = dc.trans.IRVTransformer(n_K, len(tasks), train_dataset)
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    if test:
      test_dataset = transformer.transform(test_dataset)

    # Building tensorflow IRV model
    model_irv = dc.models.TensorflowMultiTaskIRVClassifier(
        len(tasks),
        K=n_K,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

    print('-------------------------------------')
    print('Start fitting by IRV classifier')
    model_irv.fit(train_dataset, nb_epoch=nb_epoch)

    # Evaluating tensorflow IRV model
    train_scores['irv'] = model_irv.evaluate(
        train_dataset, [classification_metric], transformers)

    valid_scores['irv'] = model_irv.evaluate(
        valid_dataset, [classification_metric], transformers)

    if test:
      test_scores['irv'] = model_irv.evaluate(
          test_dataset, [classification_metric], transformers)

  if model == 'graphconv':
    # Initialize model folder

    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_filters = hyper_parameters['n_filters']
    n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    # Building graph convolution model
    with g.as_default():
      tf.set_random_seed(seed)
      graph_model = dc.nn.SequentialGraph(n_features)
      graph_model.add(dc.nn.GraphConv(int(n_filters), activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphPool())
      graph_model.add(dc.nn.GraphConv(int(n_filters), activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphPool())
      # Gather Projection
      graph_model.add(
          dc.nn.Dense(int(n_fully_connected_nodes), activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))
      with tf.Session() as sess:
        model_graphconv = dc.models.MultitaskGraphClassifier(
            sess,
            graph_model,
            len(tasks),
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_type="adam",
            beta1=.9,
            beta2=.999)

        print('-------------------------------------')
        print('Start fitting by graph convolution')
        # Fit trained model
        model_graphconv.fit(train_dataset, nb_epoch=nb_epoch)
        # Evaluating graph convolution model
        train_scores['graphconv'] = model_graphconv.evaluate(
            train_dataset, [classification_metric], transformers)

        valid_scores['graphconv'] = model_graphconv.evaluate(
            valid_dataset, [classification_metric], transformers)

        if test:
          test_scores['graphconv'] = model_graphconv.evaluate(
              test_dataset, [classification_metric], transformers)

  if model == 'rf':
    # Initialize model folder

    # Loading hyper parameters
    n_estimators = hyper_parameters['n_estimators']

    # Building scikit random forest model
    def model_builder(model_dir_rf):
      sklearn_model = RandomForestClassifier(
          class_weight="balanced", n_estimators=n_estimators, n_jobs=-1)
      return dc.models.sklearn_models.SklearnModel(sklearn_model, model_dir_rf)

    model_rf = dc.models.multitask.SingletaskToMultitask(tasks, model_builder)

    print('-------------------------------------')
    print('Start fitting by random forest')
    model_rf.fit(train_dataset)

    # Evaluating scikit random forest model
    train_scores['rf'] = model_rf.evaluate(
        train_dataset, [classification_metric], transformers)

    valid_scores['rf'] = model_rf.evaluate(
        valid_dataset, [classification_metric], transformers)

    if test:
      test_scores['rf'] = model_rf.evaluate(
          test_dataset, [classification_metric], transformers)

  return train_scores, valid_scores, test_scores


def benchmark_regression(train_dataset,
                         valid_dataset,
                         test_dataset,
                         tasks,
                         transformers,
                         hyper_parameters,
                         n_features,
                         metric='r2',
                         model='tf_regression',
                         seed=123,
                         test=False):
  """
  Calculate performance of different models on the specific dataset & tasks
  
  Parameters
  ----------
  train_dataset: dataset struct
      loaded dataset using load_* or splitter function
  valid_dataset: dataset struct
      loaded dataset using load_* or splitter function
  tasks: list of string
      list of targets(tasks, datasets)
  transformers: BalancingTransformer struct
      loaded properties of dataset from load_* function
  hyper_parameters: dict
      hyper parameters including layer size, dropout, learning rate, etc.
  n_features: integer
      number of features, or length of binary fingerprints
  model: string,  optional (default='tf_regression')
      choice of which model to use, should be: tf_regression, graphconvreg
  test: boolean
      whether to calculate test_set performance  

  Returns
  -------
  train_scores: dict
      predicting results(R2 or mae) on training set
  valid_scores: dict
      predicting results(R2 or mae) on valid set  
  test_scores : dict
	predicting results(R2 or mae) on test set
 
  """
  train_scores = {}
  valid_scores = {}
  test_scores = {}

  # Initialize metrics
  if metric == 'r2':
    regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
  elif metric == 'mae':
    regression_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                                          np.mean)

  assert model in ['tf_regression', 'graphconvreg']

  if model == 'tf_regression':
    # Loading hyper parameters
    layer_sizes = hyper_parameters['layer_sizes']
    weight_init_stddevs = hyper_parameters['weight_init_stddevs']
    bias_init_consts = hyper_parameters['bias_init_consts']
    dropouts = hyper_parameters['dropouts']
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']

    # Building tensorflow MultiTaskDNN model
    model_tf_regression = dc.models.TensorflowMultiTaskRegressor(
        len(tasks),
        n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

    print('-----------------------------------------')
    print('Start fitting by multitask DNN regression')
    model_tf_regression.fit(train_dataset, nb_epoch=nb_epoch)

    # Evaluating tensorflow MultiTaskDNN model
    train_scores['tf_regression'] = model_tf_regression.evaluate(
        train_dataset, [regression_metric], transformers)

    valid_scores['tf_regression'] = model_tf_regression.evaluate(
        valid_dataset, [regression_metric], transformers)

    if test:
      test_scores['tf_regression'] = model_tf_regression.evaluate(
          test_dataset, [regression_metric], transformers)

  if model == 'graphconvreg':
    # Initialize model folder

    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_filters = hyper_parameters['n_filters']
    n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

    g = tf.Graph()
    sess = tf.Session(graph=g)
    K.set_session(sess)
    # Building graph convoluwtion model
    with g.as_default():
      tf.set_random_seed(seed)
      graph_model = dc.nn.SequentialGraph(n_features)
      graph_model.add(dc.nn.GraphConv(int(n_filters), activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphPool())
      graph_model.add(dc.nn.GraphConv(int(n_filters), activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphPool())
      # Gather Projection
      graph_model.add(
          dc.nn.Dense(int(n_fully_connected_nodes), activation='relu'))
      graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
      graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))
      with tf.Session() as sess:
        model_graphconvreg = dc.models.MultitaskGraphRegressor(
            sess,
            graph_model,
            len(tasks),
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_type="adam",
            beta1=.9,
            beta2=.999)

        print('-------------------------------------')
        print('Start fitting by graph convolution')
        # Fit trained model
        model_graphconvreg.fit(train_dataset, nb_epoch=nb_epoch)
        # Evaluating graph convolution model
        train_scores['graphconvreg'] = model_graphconvreg.evaluate(
            train_dataset, [regression_metric], transformers)

        valid_scores['graphconvreg'] = model_graphconvreg.evaluate(
            valid_dataset, [regression_metric], transformers)

        if test:
          test_scores['graphconvreg'] = model_graphconvreg.evaluate(
              test_dataset, [regression_metric], transformers)

  return train_scores, valid_scores, test_scores


if __name__ == '__main__':
  # Global variables
  np.random.seed(123)

  parser = argparse.ArgumentParser(
      description='Deepchem benchmark: ' +
      'giving performances of different learning models on datasets')
  parser.add_argument(
      '-s',
      action='append',
      dest='splitter_args',
      default=[],
      help='Choice of splitting function: index, random, scaffold')
  parser.add_argument(
      '-m',
      action='append',
      dest='model_args',
      default=[],
      help='Choice of model: tf, tf_robust, logreg, rf, irv, graphconv, ' +
      'tf_regression, graphconvreg')
  parser.add_argument(
      '-d',
      action='append',
      dest='dataset_args',
      default=[],
      help='Choice of dataset: tox21, sider, muv, toxcast, pcba, ' +
      'kaggle, delaney, nci, pdbbind, chembl, sampl, qm7, clintox')
  parser.add_argument(
      '-t',
      action='store_true',
      dest='test',
      default=False,
      help='Evalute performance on test set')

  args = parser.parse_args()
  #Datasets and models used in the benchmark test
  splitters = args.splitter_args
  models = args.model_args
  datasets = args.dataset_args
  test = args.test

  if len(splitters) == 0:
    splitters = ['index', 'random', 'scaffold']
  if len(models) == 0:
    models = [
        'tf', 'tf_robust', 'logreg', 'graphconv', 'tf_regression',
        'graphconvreg'
    ]
    #irv and rf should be assigned manually
  if len(datasets) == 0:
    datasets = [
        'tox21', 'sider', 'muv', 'toxcast', 'pcba', 'clintox', 'sampl',
        'delaney', 'nci', 'kaggle', 'pdbbind', 'chembl', 'qm7'
    ]

  #input hyperparameters
  #tf: dropouts, learning rate, layer_sizes, weight initial stddev,penalty,
  #    batch_size
  hps = {}
  hps = {}
  hps['tf'] = [{
      'layer_sizes': [1500],
      'weight_init_stddevs': [0.02],
      'bias_init_consts': [1.],
      'dropouts': [0.5],
      'penalty': 0.1,
      'penalty_type': 'l2',
      'batch_size': 50,
      'nb_epoch': 10,
      'learning_rate': 0.001
  }]

  hps['tf_robust'] = [{
      'layer_sizes': [1500],
      'weight_init_stddevs': [0.02],
      'bias_init_consts': [1.],
      'dropouts': [0.5],
      'bypass_layer_sizes': [200],
      'bypass_weight_init_stddevs': [0.02],
      'bypass_bias_init_consts': [1.],
      'bypass_dropouts': [0.5],
      'penalty': 0.1,
      'penalty_type': 'l2',
      'batch_size': 50,
      'nb_epoch': 10,
      'learning_rate': 0.0005
  }]

  hps['logreg'] = [{
      'penalty': 0.1,
      'penalty_type': 'l2',
      'batch_size': 50,
      'nb_epoch': 10,
      'learning_rate': 0.005
  }]

  hps['irv'] = [{
      'penalty': 0.,
      'penalty_type': 'l2',
      'batch_size': 50,
      'nb_epoch': 10,
      'learning_rate': 0.001,
      'n_K': 10
  }]

  hps['graphconv'] = [{
      'batch_size': 50,
      'nb_epoch': 15,
      'learning_rate': 0.0005,
      'n_filters': 64,
      'n_fully_connected_nodes': 128,
      'seed': 123
  }]

  hps['rf'] = [{'n_estimators': 500}]

  hps['tf_regression'] = [{
      'layer_sizes': [1000, 1000],
      'weight_init_stddevs': [0.02, 0.02],
      'bias_init_consts': [1., 1.],
      'dropouts': [0.25, 0.25],
      'penalty': 0.0005,
      'penalty_type': 'l2',
      'batch_size': 128,
      'nb_epoch': 50,
      'learning_rate': 0.0008
  }]

  hps['graphconvreg'] = [{
      'batch_size': 128,
      'nb_epoch': 20,
      'learning_rate': 0.0005,
      'n_filters': 128,
      'n_fully_connected_nodes': 256,
      'seed': 123
  }]

  for split in splitters:
    for dataset in datasets:
      if dataset in ['tox21', 'sider', 'muv', 'toxcast', 'pcba', 'clintox']:
        for model in models:
          if model in ['tf', 'tf_robust', 'logreg', 'graphconv', 'rf', 'irv']:
            benchmark_loading_datasets(
                hps,
                dataset=dataset,
                model=model,
                split=split,
                out_path='.',
                test=test)
      else:
        for model in models:
          if model in ['tf_regression', 'graphconvreg']:
            benchmark_loading_datasets(
                hps,
                dataset=dataset,
                model=model,
                split=split,
                out_path='.',
                test=test)
