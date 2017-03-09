# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:25:40 2017

@author: Zhenqin Wu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import time
import csv
import numpy as np
import tensorflow as tf
import deepchem
from deepchem.molnet.run_benchmark_models import benchmark_classification, benchmark_regression


def run_benchmark(datasets,
                  model,
                  split=None,
                  metric=None,
                  featurizer=None,
                  n_features=0,
                  out_path='.',
                  test=False):
  """
  Run benchmark test on designated datasets with deepchem(or user-defined) model
  
  Parameters
  ----------
  datasets: list of string
      choice of which datasets to use, should be: tox21, muv, sider, 
      toxcast, pcba, delaney, kaggle, nci, clintox, hiv, pdbbind, chembl,
      qm7, qm7b, qm9, sampl
  model: string or user-defined model stucture
      choice of which model to use, deepchem provides implementation of
      logistic regression, random forest, multitask network, 
      bypass multitask network, irv, graph convolution;
      for user define model, it should include function: fit, evaluate
  split: string,  optional (default=None)
      choice of splitter function, None = using the default splitter
  metric: string,  optional (default=None)
      choice of evaluation metrics, None = using the default metrics(AUC & R2)
  featurizer: string or dc.feat.Featurizer,  optional (default=None)
      choice of featurization, None = using the default corresponding to model
      (string only applicable to deepchem models)
  n_features: int, optional(default=0)
      depending on featurizers, redefined when using deepchem featurizers,
      need to be specified for user-defined featurizers(if using deepchem models)
  out_path: string, optional(default='.')
      path of result file
  test: boolean, optional(default=False)
      whether to evaluate on test set
  """
  for dataset in datasets:
    if dataset in [
        'muv', 'pcba', 'tox21', 'sider', 'toxcast', 'clintox', 'hiv'
    ]:
      mode = 'classification'
      if metric == None:
        metric = [
            deepchem.metrics.Metric(deepchem.metrics.roc_auc_score, np.mean)
        ]
    elif dataset in [
        'kaggle', 'delaney', 'nci', 'pdbbind', 'chembl', 'qm7', 'qm7b', 'qm9',
        'sampl'
    ]:
      mode = 'regression'
      if metric == None:
        metric = [
            deepchem.metrics.Metric(deepchem.metrics.pearson_r2_score, np.mean)
        ]
    else:
      raise ValueError('Dataset not supported')

    if featurizer == None:
      # Assigning featurizer if not user defined
      if model in ['graphconv', 'graphconvreg']:
        featurizer = 'GraphConv'
        n_features = 75
      elif model in [
          'tf', 'tf_robust', 'logreg', 'rf', 'irv', 'tf_regression',
          'rf_regression'
      ]:
        featurizer = 'ECFP'
        n_features = 1024
      else:
        raise ValueError(
            'featurization should be specified for user-defined models')
      # Some exceptions in datasets
      if dataset in ['kaggle']:
        featurizer = None  # kaggle dataset is already featurized
        if isinstance(model,
                      str) and not model in ['tf_regression', 'rf_regression']:
          return
        if split in ['scaffold', 'butina', 'random']:
          return
      elif dataset in ['qm7', 'qm7b', 'qm9']:
        featurizer = None  # qm* datasets are already featurized
        if isinstance(model, str) and not model in ['tf_regression']:
          return
        elif model in ['tf_regression']:
          model = 'tf_regression_ft'
        if split in ['scaffold', 'butina']:
          return
      elif dataset in ['pdbbind']:
        featurizer = 'grid'  # pdbbind accepts grid featurizer
        if isinstance(model,
                      str) and not model in ['tf_regression', 'rf_regression']:
          return
        if split in ['scaffold', 'butina']:
          return

    if not split in [
        None, 'index', 'random', 'scaffold', 'butina', 'stratified'
    ]:
      raise ValueError('Splitter function not supported')

    loading_functions = {
        'tox21': deepchem.molnet.load_tox21,
        'muv': deepchem.molnet.load_muv,
        'pcba': deepchem.molnet.load_pcba,
        'nci': deepchem.molnet.load_nci,
        'sider': deepchem.molnet.load_sider,
        'toxcast': deepchem.molnet.load_toxcast,
        'kaggle': deepchem.molnet.load_kaggle,
        'delaney': deepchem.molnet.load_delaney,
        'pdbbind': deepchem.molnet.load_pdbbind_grid,
        'chembl': deepchem.molnet.load_chembl,
        'qm7': deepchem.molnet.load_qm7_from_mat,
        'qm7b': deepchem.molnet.load_qm7b_from_mat,
        'qm9': deepchem.molnet.load_qm9,
        'sampl': deepchem.molnet.load_sampl,
        'clintox': deepchem.molnet.load_clintox,
        'hiv': deepchem.molnet.load_hiv
    }

    print('-------------------------------------')
    print('Benchmark on dataset: %s' % dataset)
    print('-------------------------------------')
    # loading datasets
    if split is not None:
      print('Splitting function: %s' % split)
      tasks, all_dataset, transformers = loading_functions[dataset](
          featurizer=featurizer, split=split)
    else:
      tasks, all_dataset, transformers = loading_functions[dataset](
          featurizer=featurizer)

    train_dataset, valid_dataset, test_dataset = all_dataset
    if dataset in ['kaggle', 'pdbbind']:
      n_features = train_dataset.get_data_shape()[0]
    elif dataset in ['qm7', 'qm7b', 'qm9']:
      n_features = list(train_dataset.get_data_shape())

    time_start_fitting = time.time()
    train_scores = {}
    valid_scores = {}
    test_scores = {}

    if isinstance(model, str):
      if mode == 'classification':
        train_score, valid_score, test_score = benchmark_classification(
            train_dataset,
            valid_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
            model=model,
            test=test)
      elif mode == 'regression':
        train_score, valid_score, test_score = benchmark_regression(
            train_dataset,
            valid_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
            model=model,
            test=test)
    else:
      model.fit(train_dataset)
      train_scores['user_defined'] = model.evaluate(train_dataset, metric,
                                                    transformers)
      valid_scores['user_defined'] = model.evaluate(valid_dataset, metric,
                                                    transformers)
      if test:
        test_scores['user_defined'] = model.evaluate(test_dataset, metric,
                                                     transformers)

    time_finish_fitting = time.time()

    with open(os.path.join(out_path, 'results.csv'), 'a') as f:
      writer = csv.writer(f)
      for i in train_score:
        output_line = [
            dataset, str(split), mode, 'train', i,
            train_score[i][list(train_score[i].keys())[0]], 'valid', i,
            valid_score[i][list(valid_score[i].keys())[0]]
        ]
        if test:
          output_line.extend(
              ['test', i, test_score[i][list(test_score[i].keys(0))[0]]])
        output_line.extend(
            ['time_for_running', time_finish_fitting - time_start_fitting])
        writer.writerow(output_line)
