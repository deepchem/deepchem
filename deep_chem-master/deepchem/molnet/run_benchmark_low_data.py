# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:25:40 2017

@author: Zhenqin Wu
"""
'''
import os
import time
import csv
import numpy as np
import tensorflow as tf
import deepchem
from deepchem.molnet.run_benchmark_models import low_data_benchmark_classification
from deepchem.molnet.check_availability import CheckFeaturizer


def run_benchmark_low_data(datasets,
                           model,
                           split='task',
                           metric=None,
                           featurizer=None,
                           n_features=0,
                           out_path='.',
                           K=4,
                           hyper_parameters=None,
                           cross_valid=False,
                           seed=123):
  """
  Run low data benchmark test on designated datasets 
  with deepchem(or user-defined) model
  
  Parameters
  ----------
  datasets: list of string
      choice of which datasets to use, should be: muv, tox21, sider 
  model: string or user-defined model stucture
      choice of which model to use, should be: siamese, attn, res
  split: string,  optional (default='task')
      choice of splitter function, only task splitter supported
  metric: string,  optional (default=None)
      choice of evaluation metrics, None = using the default metrics(AUC)
  featurizer: string or dc.feat.Featurizer,  optional (default=None)
      choice of featurization, None = using the default corresponding to model
      (string only applicable to deepchem models)
  n_features: int, optional(default=0)
      depending on featurizers, redefined when using deepchem featurizers,
      need to be specified for user-defined featurizers(if using deepchem models)
  out_path: string, optional(default='.')
      path of result file
  K: int, optional(default=4)
      K-fold splitting of datasets
  hyper_parameters: dict, optional (default=None)
      hyper parameters for designated model, None = use preset values
  cross_valid: boolean, optional(default=False)
      whether to cross validate
  """
  for dataset in datasets:
    if dataset in ['muv', 'sider', 'tox21']:
      mode = 'classification'
      if metric == None:
        metric = str('auc')
    else:
      raise ValueError('Dataset not supported')

    metric_all = {
        'auc': deepchem.metrics.Metric(deepchem.metrics.roc_auc_score, np.mean)
    }

    if isinstance(metric, str):
      metric = metric_all[metric]

    if featurizer == None and isinstance(model, str):
      # Assigning featurizer if not user defined
      pair = (dataset, model)
      if pair in CheckFeaturizer:
        featurizer = CheckFeaturizer[pair][0]
        n_features = CheckFeaturizer[pair][1]
      else:
        continue

    loading_functions = {
        'muv': deepchem.molnet.load_muv,
        'sider': deepchem.molnet.load_sider,
        'tox21': deepchem.molnet.load_tox21
    }
    assert split == 'task'

    print('-------------------------------------')
    print('Benchmark on dataset: %s' % dataset)
    print('-------------------------------------')
    # loading datasets
    print('Splitting function: %s' % split)
    tasks, all_dataset, transformers = loading_functions[dataset](
        featurizer=featurizer, split=split, K=K)

    if cross_valid:
      num_iter = K  # K iterations for cross validation
    else:
      num_iter = 1
    for count_iter in range(num_iter):
      # Assembling train and valid datasets
      train_folds = all_dataset[:K - count_iter - 1] + all_dataset[K -
                                                                   count_iter:]
      train_dataset = deepchem.splits.merge_fold_datasets(train_folds)
      valid_dataset = all_dataset[K - count_iter - 1]

      time_start_fitting = time.time()
      train_score = {}
      valid_score = {}

      if isinstance(model, str):
        if mode == 'classification':
          valid_score = low_data_benchmark_classification(
              train_dataset,
              valid_dataset,
              n_features,
              metric,
              model=model,
              hyper_parameters=hyper_parameters,
              seed=seed)
      else:
        model.fit(train_dataset)
        valid_score['user_defined'] = model.evaluate(valid_dataset, metric,
                                                     transformers)

      time_finish_fitting = time.time()

      with open(os.path.join(out_path, 'results.csv'), 'a') as f:
        writer = csv.writer(f)
        for i in valid_score:
          output_line = [dataset, str(split), mode, 'valid', i]
          for task in valid_score[i][0]:
            output_line.extend(
                [task, valid_score[i][0][task], valid_score[i][1][task]])
          output_line.extend(
              ['time_for_running', time_finish_fitting - time_start_fitting])
          writer.writerow(output_line)
'''
