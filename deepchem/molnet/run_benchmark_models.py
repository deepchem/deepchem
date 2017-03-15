#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:41:26 2017

@author: zqwu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem
from deepchem.molnet.preset_hyper_parameters import hps
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def benchmark_classification(
    train_dataset,
    valid_dataset,
    test_dataset,
    tasks,
    transformers,
    n_features,
    metric,
    model,
    test=False,
    hyper_parameters=None,
    seed=123,):
  """
  Calculate performance of different models on the specific dataset & tasks
  
  Parameters
  ----------
  train_dataset: dataset struct
      dataset used for model training and evaluation
  valid_dataset: dataset struct
      dataset only used for model evaluation (and hyperparameter tuning)
  test_dataset: dataset struct
      dataset only used for model evaluation
  tasks: list of string
      list of targets(tasks, datasets)
  transformers: dc.trans.Transformer struct
      transformer used for model evaluation
  n_features: integer
      number of features, or length of binary fingerprints
  metric: list of dc.metrics.Metric objects
      metrics used for evaluation
  model: string,  optional (default='tf')
      choice of which model to use, should be: rf, tf, tf_robust, logreg,
      irv, graphconv
  test: boolean
      whether to calculate test_set performance
  hyper_parameters: dict, optional (default=None)
      hyper parameters for designated model, None = use preset values
  

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

  assert model in ['rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv']
  if hyper_parameters is None:
    hyper_parameters = hps[model]
  model_name = model

  if model_name == 'tf':
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
    model = deepchem.models.TensorflowMultiTaskClassifier(
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

  elif model_name == 'tf_robust':
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
    model = deepchem.models.RobustMultitaskClassifier(
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

  elif model_name == 'logreg':
    # Loading hyper parameters
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']

    # Building tensorflow logistic regression model
    model = deepchem.models.TensorflowLogisticRegression(
        len(tasks),
        n_features,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

  elif model_name == 'irv':
    # Loading hyper parameters
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_K = hyper_parameters['n_K']

    # Transform fingerprints to IRV features
    transformer = deepchem.trans.IRVTransformer(n_K, len(tasks), train_dataset)
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    if test:
      test_dataset = transformer.transform(test_dataset)

    # Building tensorflow IRV model
    model = deepchem.models.TensorflowMultiTaskIRVClassifier(
        len(tasks),
        K=n_K,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

  elif model_name == 'graphconv':
    # Initialize model folder

    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_filters = hyper_parameters['n_filters']
    n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

    tf.set_random_seed(seed)
    graph_model = deepchem.nn.SequentialGraph(n_features)
    graph_model.add(
        deepchem.nn.GraphConv(int(n_filters), n_features, activation='relu'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(deepchem.nn.GraphPool())
    graph_model.add(
        deepchem.nn.GraphConv(
            int(n_filters), int(n_filters), activation='relu'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(deepchem.nn.GraphPool())
    # Gather Projection
    graph_model.add(
        deepchem.nn.Dense(
            int(n_fully_connected_nodes), int(n_filters), activation='relu'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(deepchem.nn.GraphGather(batch_size, activation="tanh"))
    model = deepchem.models.MultitaskGraphClassifier(
        graph_model,
        len(tasks),
        n_features,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

  elif model_name == 'rf':
    # Loading hyper parameters
    n_estimators = hyper_parameters['n_estimators']
    nb_epoch = None

    # Building scikit random forest model
    def model_builder(model_dir_rf):
      sklearn_model = RandomForestClassifier(
          class_weight="balanced", n_estimators=n_estimators, n_jobs=-1)
      return deepchem.models.sklearn_models.SklearnModel(sklearn_model,
                                                         model_dir_rf)

    model = deepchem.models.multitask.SingletaskToMultitask(tasks,
                                                            model_builder)

  if nb_epoch is None:
    model.fit(train_dataset)
  else:
    model.fit(train_dataset, nb_epoch=nb_epoch)

  train_scores[model_name] = model.evaluate(train_dataset, metric, transformers)
  valid_scores[model_name] = model.evaluate(valid_dataset, metric, transformers)
  if test:
    test_scores[model_name] = model.evaluate(test_dataset, metric, transformers)

  return train_scores, valid_scores, test_scores


def benchmark_regression(
    train_dataset,
    valid_dataset,
    test_dataset,
    tasks,
    transformers,
    n_features,
    metric,
    model,
    test=False,
    hyper_parameters=None,
    seed=123,):
  """
  Calculate performance of different models on the specific dataset & tasks
  
  Parameters
  ----------
  train_dataset: dataset struct
      dataset used for model training and evaluation
  valid_dataset: dataset struct
      dataset only used for model evaluation (and hyperparameter tuning)
  test_dataset: dataset struct
      dataset only used for model evaluation
  tasks: list of string
      list of targets(tasks, datasets)
  transformers: dc.trans.Transformer struct
      transformer used for model evaluation
  n_features: integer
      number of features, or length of binary fingerprints
  metric: list of dc.metrics.Metric objects
      metrics used for evaluation
  model: string,  optional (default='tf_regression')
      choice of which model to use, should be: tf_regression, tf_regression_ft,
      graphconvreg, rf_regression
  test: boolean
      whether to calculate test_set performance
  hyper_parameters: dict, optional (default=None)
      hyper parameters for designated model, None = use preset values
  

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

  assert model in [
      'tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg'
  ]
  if hyper_parameters is None:
    hyper_parameters = hps[model]
  model_name = model

  if model_name == 'tf_regression':
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

    model = deepchem.models.TensorflowMultiTaskRegressor(
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

    # Building tensorflow MultiTaskDNN model
  elif model_name == 'tf_regression_ft':
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
    fit_transformers = [hyper_parameters['fit_transformers'](train_dataset)]

    model = deepchem.models.TensorflowMultiTaskFitTransformRegressor(
        n_tasks=len(tasks),
        n_features=n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        penalty=penalty,
        penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fit_transformers=fit_transformers,
        n_eval=10,
        seed=seed)

  elif model_name == 'graphconvreg':
    # Initialize model folder

    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_filters = hyper_parameters['n_filters']
    n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

    tf.set_random_seed(seed)
    graph_model = deepchem.nn.SequentialGraph(n_features)
    graph_model.add(
        deepchem.nn.GraphConv(int(n_filters), n_features, activation='relu'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(deepchem.nn.GraphPool())
    graph_model.add(
        deepchem.nn.GraphConv(
            int(n_filters), int(n_filters), activation='relu'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(deepchem.nn.GraphPool())
    # Gather Projection
    graph_model.add(
        deepchem.nn.Dense(
            int(n_fully_connected_nodes), int(n_filters), activation='relu'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(deepchem.nn.GraphGather(batch_size, activation="tanh"))
    model = deepchem.models.MultitaskGraphRegressor(
        graph_model,
        len(tasks),
        n_features,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

  elif model_name == 'DTNN':
    # Initialize model folder

    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_distance = hyper_parameters['n_distance']
    n_hidden = hyper_parameters['n_hidden']

    tf.set_random_seed(seed)
    graph_model = dc.nn.SequentialDTNNGraph(max_n_atoms=n_features[0], 
                                            n_distance=n_distance)
    graph_model.add(dc.nn.DTNNEmbedding(n_features=n_hidden))
    graph_model.add(dc.nn.DTNNStep(n_features=n_hidden, n_distance=n_distance))
    graph_model.add(dc.nn.DTNNStep(n_features=n_hidden, n_distance=n_distance))
    graph_model.add(dc.nn.DTNNGather(n_tasks=len(tasks)))

    model = dc.models.DTNNRegressor(
        graph_model,
        n_tasks=len(tasks),
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

  elif model_name == 'rf_regression':
    # Loading hyper parameters
    n_estimators = hyper_parameters['n_estimators']
    nb_epoch = None

    # Building scikit random forest model
    def model_builder(model_dir_rf_regression):
      sklearn_model = RandomForestRegressor(
          n_estimators=n_estimators, n_jobs=-1)
      return deepchem.models.sklearn_models.SklearnModel(
          sklearn_model, model_dir_rf_regression)

    model = deepchem.models.multitask.SingletaskToMultitask(tasks,
                                                            model_builder)

  print('-----------------------------')
  print('Start fitting: %s' % model_name)
  if nb_epoch is None:
    model.fit(train_dataset)
  else:
    model.fit(train_dataset, nb_epoch=nb_epoch)

  train_scores[model_name] = model.evaluate(train_dataset, metric, transformers)
  valid_scores[model_name] = model.evaluate(valid_dataset, metric, transformers)
  if test:
    test_scores[model_name] = model.evaluate(test_dataset, metric, transformers)

  return train_scores, valid_scores, test_scores
