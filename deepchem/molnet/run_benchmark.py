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
from deepchem.molnet.check_availability import CheckFeaturizer, CheckSplit


def run_benchmark(datasets,
                  model,
                  split=None,
                  metric=None,
                  featurizer=None,
                  n_features=0,
                  out_path='.',
                  hyper_parameters=None,
                  test=False,
                  seed=123):
  """
  Run benchmark test on designated datasets with deepchem(or user-defined) model
  
  Parameters
  ----------
  datasets: list of string
      choice of which datasets to use, should be: bace_c, bace_r, bbbp, chembl,
      clearance, clintox, delaney, hiv, hopv, kaggle, lipo, muv, nci, pcba, 
      pdbbind, ppb, qm7, qm7b, qm8, qm9, sampl, sider, tox21, toxcast 
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
  hyper_parameters: dict, optional (default=None)
      hyper parameters for designated model, None = use preset values
  test: boolean, optional(default=False)
      whether to evaluate on test set
  """
  for dataset in datasets:
    if dataset in [
        'bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'sider', 'tox21',
        'toxcast'
    ]:
      mode = 'classification'
      if metric == None:
        metric = str('auc')
    elif dataset in [
        'bace_r', 'chembl', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo',
        'nci', 'pdbbind', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl'
    ]:
      mode = 'regression'
      if metric == None:
        metric = str('r2'）
    else:
      raise ValueError('Dataset not supported')

    metric_all = {
        'auc': deepchem.metrics.Metric(deepchem.metrics.roc_auc_score, np.mean),
        'r2':
        deepchem.metrics.Metric(deepchem.metrics.pearson_r2_score, np.mean)
    }

    if isinstance(metric, str):
      metric = [metric_all[metric]]

    if featurizer == None and isinstance(model, str):
      # Assigning featurizer if not user defined
      pair = (dataset, model)
      if pair in CheckFeaturizer:
        featurizer = CheckFeaturizer[pair][0]
        n_features = CheckFeaturizer[pair][1]
      else:
        continue

    if not split in [None] + CheckSplit[dataset]:
      continue

    loading_functions = {
        'bace_c': deepchem.molnet.load_bace_classification,
        'bace_r': deepchem.molnet.load_bace_regression,
        'bbbp': deepchem.molnet.load_bbbp,
        'chembl': deepchem.molnet.load_chembl,
        'clearance': deepchem.molnet.load_clearance,
        'clintox': deepchem.molnet.load_clintox,
        'delaney': deepchem.molnet.load_delaney,
        'hiv': deepchem.molnet.load_hiv,
        'hopv': deepchem.molnet.load_hopv,
        'kaggle': deepchem.molnet.load_kaggle,
        'lipo': deepchem.molnet.load_lipo,
        'muv': deepchem.molnet.load_muv,
        'nci': deepchem.molnet.load_nci,
        'pcba': deepchem.molnet.load_pcba,
        'pdbbind': deepchem.molnet.load_pdbbind_grid,
        'ppb': deepchem.molnet.load_ppb,
        'qm7': deepchem.molnet.load_qm7_from_mat,
        'qm7b': deepchem.molnet.load_qm7b_from_mat,
        'qm8': deepchem.molnet.load_qm8,
        'qm9': deepchem.molnet.load_qm9,
        'sampl': deepchem.molnet.load_sampl,
        'sider': deepchem.molnet.load_sider,
        'tox21': deepchem.molnet.load_tox21,
        'toxcast': deepchem.molnet.load_toxcast
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

    time_start_fitting = time.time()
    train_score = {}
    valid_score = {}
    test_score = {}

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
            model,
            test=test,
            hyper_parameters=hyper_parameters,
            seed=seed)
      elif mode == 'regression':
        train_score, valid_score, test_score = benchmark_regression(
            train_dataset,
            valid_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
            model,
            test=test,
            hyper_parameters=hyper_parameters,
            seed=seed)
    else:
      model.fit(train_dataset)
      train_score['user_defined'] = model.evaluate(train_dataset, metric,
                                                   transformers)
      valid_score['user_defined'] = model.evaluate(valid_dataset, metric,
                                                   transformers)
      if test:
        test_score['user_defined'] = model.evaluate(test_dataset, metric,
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
              ['test', i, test_score[i][list(test_score[i].keys())[0]]])
        output_line.extend(
            ['time_for_running', time_finish_fitting - time_start_fitting])
        writer.writerow(output_line)
