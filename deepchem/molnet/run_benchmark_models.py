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
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge


def benchmark_classification(train_dataset,
                             valid_dataset,
                             test_dataset,
                             tasks,
                             transformers,
                             n_features,
                             metric,
                             model,
                             test=False,
                             hyper_parameters=None,
                             seed=123):
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
      irv, graphconv, dag, xgb, weave
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
      'rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv', 'dag', 'xgb',
      'weave', 'kernelsvm'
  ]
  if hyper_parameters is None:
    hyper_parameters = hps[model]
  model_name = model
  import xgboost

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

  elif model_name == 'dag':
    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    default_max_atoms = hyper_parameters['default_max_atoms']

    max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
    max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
    max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
    max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])
    max_atoms = min([max_atoms, default_max_atoms])
    print('Maximum number of atoms: %i' % max_atoms)
    reshard_size = 256
    transformer = deepchem.trans.DAGTransformer(max_atoms=max_atoms)
    train_dataset.reshard(reshard_size)
    train_dataset = transformer.transform(train_dataset)
    valid_dataset.reshard(reshard_size)
    valid_dataset = transformer.transform(valid_dataset)
    if test:
      test_dataset.reshard(reshard_size)
      test_dataset = transformer.transform(test_dataset)

    tf.set_random_seed(seed)
    graph_model = deepchem.nn.SequentialDAGGraph(
        n_features, max_atoms=max_atoms)
    graph_model.add(
        deepchem.nn.DAGLayer(
            n_graph_feat,
            n_features,
            max_atoms=max_atoms,
            batch_size=batch_size))
    graph_model.add(deepchem.nn.DAGGather(n_graph_feat, max_atoms=max_atoms))

    model = deepchem.models.MultitaskGraphClassifier(
        graph_model,
        len(tasks),
        n_features,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

  elif model_name == 'weave':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    n_pair_feat = hyper_parameters['n_pair_feat']

    max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
    max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
    max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
    max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])

    tf.set_random_seed(seed)
    graph_model = deepchem.nn.AlternateSequentialWeaveGraph(
        batch_size,
        max_atoms=max_atoms,
        n_atom_feat=n_features,
        n_pair_feat=n_pair_feat)
    graph_model.add(deepchem.nn.AlternateWeaveLayer(max_atoms, 75, 14))
    graph_model.add(
        deepchem.nn.AlternateWeaveLayer(max_atoms, 50, 50, update_pair=False))
    graph_model.add(deepchem.nn.Dense(n_graph_feat, 50, activation='tanh'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(
        deepchem.nn.AlternateWeaveGather(
            batch_size, n_input=n_graph_feat, gaussian_expand=True))

    model = deepchem.models.MultitaskGraphClassifier(
        graph_model,
        len(tasks),
        n_features,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_decay_time=1000,
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
      return deepchem.models.sklearn_models.SklearnModel(
          sklearn_model, model_dir_rf)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'kernelsvm':
    # Loading hyper parameters
    C = hyper_parameters['C']
    gamma = hyper_parameters['gamma']
    nb_epoch = None

    # Building scikit learn Kernel SVM model
    def model_builder(model_dir_kernelsvm):
      sklearn_model = SVC(
          C=C, gamma=gamma, class_weight="balanced", probability=True)
      return deepchem.models.SklearnModel(sklearn_model, model_dir_kernelsvm)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'xgb':
    # Loading hyper parameters
    max_depth = hyper_parameters['max_depth']
    learning_rate = hyper_parameters['learning_rate']
    n_estimators = hyper_parameters['n_estimators']
    gamma = hyper_parameters['gamma']
    min_child_weight = hyper_parameters['min_child_weight']
    max_delta_step = hyper_parameters['max_delta_step']
    subsample = hyper_parameters['subsample']
    colsample_bytree = hyper_parameters['colsample_bytree']
    colsample_bylevel = hyper_parameters['colsample_bylevel']
    reg_alpha = hyper_parameters['reg_alpha']
    reg_lambda = hyper_parameters['reg_lambda']
    scale_pos_weight = hyper_parameters['scale_pos_weight']
    base_score = hyper_parameters['base_score']
    seed = hyper_parameters['seed']
    early_stopping_rounds = hyper_parameters['early_stopping_rounds']
    nb_epoch = None

    esr = {'early_stopping_rounds': early_stopping_rounds}

    # Building xgboost classification model
    def model_builder(model_dir_xgb):
      xgboost_model = xgboost.XGBClassifier(
          max_depth=max_depth,
          learning_rate=learning_rate,
          n_estimators=n_estimators,
          gamma=gamma,
          min_child_weight=min_child_weight,
          max_delta_step=max_delta_step,
          subsample=subsample,
          colsample_bytree=colsample_bytree,
          colsample_bylevel=colsample_bylevel,
          reg_alpha=reg_alpha,
          reg_lambda=reg_lambda,
          scale_pos_weight=scale_pos_weight,
          base_score=base_score,
          seed=seed)
      return deepchem.models.xgboost_models.XGBoostModel(
          xgboost_model, model_dir_xgb, **esr)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  if nb_epoch is None:
    model.fit(train_dataset)
  else:
    model.fit(train_dataset, nb_epoch=nb_epoch)

  train_scores[model_name] = model.evaluate(train_dataset, metric, transformers)
  valid_scores[model_name] = model.evaluate(valid_dataset, metric, transformers)
  if test:
    test_scores[model_name] = model.evaluate(test_dataset, metric, transformers)

  return train_scores, valid_scores, test_scores


def benchmark_regression(train_dataset,
                         valid_dataset,
                         test_dataset,
                         tasks,
                         transformers,
                         n_features,
                         metric,
                         model,
                         test=False,
                         hyper_parameters=None,
                         seed=123):
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
      graphconvreg, rf_regression, dtnn, dag_regression, xgb_regression,
      weave_regression, krr, ani, krr_ft, mpnn
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
      'tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg',
      'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression', 'krr',
      'ani', 'krr_ft', 'mpnn'
  ]
  import xgboost
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

  elif model_name == 'dtnn':
    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_embedding = hyper_parameters['n_embedding']
    n_distance = hyper_parameters['n_distance']
    assert len(n_features) == 2, 'DTNN is only applicable to qm datasets'

    tf.set_random_seed(seed)
    graph_model = deepchem.nn.SequentialDTNNGraph(n_distance=n_distance)
    graph_model.add(deepchem.nn.DTNNEmbedding(n_embedding=n_embedding))
    graph_model.add(
        deepchem.nn.DTNNStep(n_embedding=n_embedding, n_distance=n_distance))
    graph_model.add(
        deepchem.nn.DTNNStep(n_embedding=n_embedding, n_distance=n_distance))
    graph_model.add(deepchem.nn.DTNNGather(n_embedding=n_embedding))
    model = deepchem.models.MultitaskGraphRegressor(
        graph_model,
        len(tasks),
        n_embedding,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

  elif model_name == 'dag_regression':
    # Loading hyper parameters
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    default_max_atoms = hyper_parameters['default_max_atoms']

    max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
    max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
    max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
    max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])
    max_atoms = min([max_atoms, default_max_atoms])
    print('Maximum number of atoms: %i' % max_atoms)
    reshard_size = 512
    transformer = deepchem.trans.DAGTransformer(max_atoms=max_atoms)
    train_dataset.reshard(reshard_size)
    train_dataset = transformer.transform(train_dataset)
    valid_dataset.reshard(reshard_size)
    valid_dataset = transformer.transform(valid_dataset)
    if test:
      test_dataset.reshard(reshard_size)
      test_dataset = transformer.transform(test_dataset)

    tf.set_random_seed(seed)
    graph_model = deepchem.nn.SequentialDAGGraph(
        n_features, max_atoms=max_atoms)
    graph_model.add(
        deepchem.nn.DAGLayer(
            n_graph_feat,
            n_features,
            max_atoms=max_atoms,
            batch_size=batch_size))
    graph_model.add(deepchem.nn.DAGGather(n_graph_feat, max_atoms=max_atoms))

    model = deepchem.models.MultitaskGraphRegressor(
        graph_model,
        len(tasks),
        n_features,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

  elif model_name == 'weave_regression':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    n_pair_feat = hyper_parameters['n_pair_feat']

    max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
    max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
    max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
    max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])

    tf.set_random_seed(seed)
    graph_model = deepchem.nn.AlternateSequentialWeaveGraph(
        batch_size,
        max_atoms=max_atoms,
        n_atom_feat=n_features,
        n_pair_feat=n_pair_feat)
    graph_model.add(deepchem.nn.AlternateWeaveLayer(max_atoms, 75, 14))
    graph_model.add(
        deepchem.nn.AlternateWeaveLayer(max_atoms, 50, 50, update_pair=False))
    graph_model.add(deepchem.nn.Dense(n_graph_feat, 50, activation='tanh'))
    graph_model.add(deepchem.nn.BatchNormalization(epsilon=1e-5, mode=1))
    graph_model.add(
        deepchem.nn.AlternateWeaveGather(
            batch_size, n_input=n_graph_feat, gaussian_expand=True))

    model = deepchem.models.MultitaskGraphRegressor(
        graph_model,
        len(tasks),
        n_features,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_decay_time=1000,
        optimizer_type="adam",
        beta1=.9,
        beta2=.999)

  elif model_name == 'ani':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    layer_structures = hyper_parameters['layer_structures']

    assert len(n_features) == 2, 'ANI is only applicable to qm datasets'
    max_atoms = n_features[0]
    atom_number_cases = np.unique(
        np.concatenate([
            train_dataset.X[:, :, 0], valid_dataset.X[:, :, 0],
            test_dataset.X[:, :, 0]
        ]))

    atom_number_cases = atom_number_cases.astype(int).tolist()
    try:
      # Remove token for paddings
      atom_number_cases.remove(0)
    except:
      pass
    ANItransformer = deepchem.trans.ANITransformer(
        max_atoms=max_atoms, atom_cases=atom_number_cases)
    train_dataset = ANItransformer.transform(train_dataset)
    valid_dataset = ANItransformer.transform(valid_dataset)
    if test:
      test_dataset = ANItransformer.transform(test_dataset)
    n_feat = ANItransformer.get_num_feats() - 1

    model = deepchem.models.ANIRegression(
        len(tasks),
        max_atoms,
        n_feat,
        layer_structures=layer_structures,
        atom_number_cases=atom_number_cases,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        mode="regression",
        random_seed=seed)

  elif model_name == 'mpnn':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    T = hyper_parameters['T']
    M = hyper_parameters['M']

    model = deepchem.models.MPNNTensorGraph(
        len(tasks),
        n_atom_feat=n_features[0],
        n_pair_feat=n_features[1],
        n_hidden=n_features[0],
        T=T,
        M=M,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        mode="regression")

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

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'krr':
    # Loading hyper parameters
    alpha = hyper_parameters['alpha']
    nb_epoch = None

    # Building scikit learn Kernel Ridge Regression model
    def model_builder(model_dir_krr):
      sklearn_model = KernelRidge(kernel="rbf", alpha=alpha)
      return deepchem.models.SklearnModel(sklearn_model, model_dir_krr)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'krr_ft':
    # Loading hyper parameters
    alpha = hyper_parameters['alpha']
    nb_epoch = None

    ft_transformer = deepchem.trans.CoulombFitTransformer(train_dataset)
    train_dataset = ft_transformer.transform(train_dataset)
    valid_dataset = ft_transformer.transform(valid_dataset)
    test_dataset = ft_transformer.transform(test_dataset)

    # Building scikit learn Kernel Ridge Regression model
    def model_builder(model_dir_krr):
      sklearn_model = KernelRidge(kernel="rbf", alpha=alpha)
      return deepchem.models.SklearnModel(sklearn_model, model_dir_krr)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'xgb_regression':
    # Loading hyper parameters
    max_depth = hyper_parameters['max_depth']
    learning_rate = hyper_parameters['learning_rate']
    n_estimators = hyper_parameters['n_estimators']
    gamma = hyper_parameters['gamma']
    min_child_weight = hyper_parameters['min_child_weight']
    max_delta_step = hyper_parameters['max_delta_step']
    subsample = hyper_parameters['subsample']
    colsample_bytree = hyper_parameters['colsample_bytree']
    colsample_bylevel = hyper_parameters['colsample_bylevel']
    reg_alpha = hyper_parameters['reg_alpha']
    reg_lambda = hyper_parameters['reg_lambda']
    scale_pos_weight = hyper_parameters['scale_pos_weight']
    base_score = hyper_parameters['base_score']
    seed = hyper_parameters['seed']
    early_stopping_rounds = hyper_parameters['early_stopping_rounds']
    nb_epoch = None

    esr = {'early_stopping_rounds': early_stopping_rounds}

    # Building xgboost classification model
    def model_builder(model_dir_xgb):
      xgboost_model = xgboost.XGBRegressor(
          max_depth=max_depth,
          learning_rate=learning_rate,
          n_estimators=n_estimators,
          gamma=gamma,
          min_child_weight=min_child_weight,
          max_delta_step=max_delta_step,
          subsample=subsample,
          colsample_bytree=colsample_bytree,
          colsample_bylevel=colsample_bylevel,
          reg_alpha=reg_alpha,
          reg_lambda=reg_lambda,
          scale_pos_weight=scale_pos_weight,
          base_score=base_score,
          seed=seed)
      return deepchem.models.xgboost_models.XGBoostModel(
          xgboost_model, model_dir_xgb, **esr)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

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


def low_data_benchmark_classification(train_dataset,
                                      valid_dataset,
                                      n_features,
                                      metric,
                                      model='siamese',
                                      hyper_parameters=None,
                                      seed=123):
  """
  Calculate low data benchmark performance

  Parameters
  ----------
  train_dataset : dataset struct
      loaded dataset, ConvMol struct, used for training
  valid_dataset : dataset struct
      loaded dataset, ConvMol struct, used for validation
  n_features : integer
      number of features, or length of binary fingerprints
  metric: list of dc.metrics.Metric objects
      metrics used for evaluation
  model : string,  optional (default='siamese')
      choice of which model to use, should be: siamese, attn, res
  hyper_parameters: dict, optional (default=None)
      hyper parameters for designated model, None = use preset values

  Returns
  -------
  valid_scores : dict
	predicting results(AUC) on valid set

  """
  train_scores = {}  # train set not evaluated in low data model
  valid_scores = {}

  assert model in ['siamese', 'attn', 'res']
  if hyper_parameters is None:
    hyper_parameters = hps[model]

  # Loading hyperparameters
  # num positive/negative ligands
  n_pos = hyper_parameters['n_pos']
  n_neg = hyper_parameters['n_neg']
  # Set batch sizes for network
  test_batch_size = hyper_parameters['test_batch_size']
  support_batch_size = n_pos + n_neg
  # Model structure
  n_filters = hyper_parameters['n_filters']
  n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

  # Traning settings
  nb_epochs = hyper_parameters['nb_epochs']
  n_train_trials = hyper_parameters['n_train_trials']
  n_eval_trials = hyper_parameters['n_eval_trials']

  learning_rate = hyper_parameters['learning_rate']

  tf.set_random_seed(seed)
  support_graph = deepchem.nn.SequentialSupportGraph(n_features)
  prev_features = n_features
  for count, n_filter in enumerate(n_filters):
    support_graph.add(
        deepchem.nn.GraphConv(int(n_filter), prev_features, activation='relu'))
    support_graph.add(deepchem.nn.GraphPool())
    prev_features = int(n_filter)

  for count, n_fcnode in enumerate(n_fully_connected_nodes):
    support_graph.add(
        deepchem.nn.Dense(int(n_fcnode), prev_features, activation='tanh'))
    prev_features = int(n_fcnode)

  support_graph.add_test(
      deepchem.nn.GraphGather(test_batch_size, activation='tanh'))
  support_graph.add_support(
      deepchem.nn.GraphGather(support_batch_size, activation='tanh'))
  if model in ['siamese']:
    pass
  elif model in ['attn']:
    max_depth = hyper_parameters['max_depth']
    support_graph.join(
        deepchem.nn.AttnLSTMEmbedding(test_batch_size, support_batch_size,
                                      prev_features, max_depth))
  elif model in ['res']:
    max_depth = hyper_parameters['max_depth']
    support_graph.join(
        deepchem.nn.ResiLSTMEmbedding(test_batch_size, support_batch_size,
                                      prev_features, max_depth))

  model_low_data = deepchem.models.SupportGraphClassifier(
      support_graph,
      test_batch_size=test_batch_size,
      support_batch_size=support_batch_size,
      learning_rate=learning_rate)

  print('-------------------------------------')
  print('Start fitting by low data model: ' + model)
  # Fit trained model
  model_low_data.fit(
      train_dataset,
      nb_epochs=nb_epochs,
      n_episodes_per_epoch=n_train_trials,
      n_pos=n_pos,
      n_neg=n_neg,
      log_every_n_samples=50)

  # Evaluating low data model
  valid_scores[model] = model_low_data.evaluate(
      valid_dataset, metric, n_pos, n_neg, n_trials=n_eval_trials)

  return valid_scores
