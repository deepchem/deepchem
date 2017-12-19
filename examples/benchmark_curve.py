"""
Created on Sat Oct 14 16:59:49 2017

@author: zqwu

This script evaluates how performances change with
different size of training set(training set fraction).

Default fractions evaluated are 0.1, 0.2, ..., 0.9.
The whole dataset is split into train set and valid set
with corresponding fractions.(test set is not used)
Models are trained on train set and evaluated on valid set.
Command line options are the same as `benchmark.py`

All results and train set fractions are stored in
'./results_frac_train_curve.csv'
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import deepchem as dc
import argparse
import pickle
import csv
from deepchem.molnet.run_benchmark_models import benchmark_classification, benchmark_regression
from deepchem.molnet.run_benchmark import load_dataset
from deepchem.molnet.check_availability import CheckFeaturizer, CheckSplit
from deepchem.molnet.preset_hyper_parameters import hps

# Evaluate performances with different training set fraction
frac_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser(
    description='Deepchem benchmark: ' +
    'giving performances of different learning models on datasets')
parser.add_argument(
    '-s',
    action='append',
    dest='splitter_args',
    default=[],
    help='Choice of splitting function: index, random, scaffold, stratified')
parser.add_argument(
    '-m',
    action='append',
    dest='model_args',
    default=[],
    help='Choice of model: tf, tf_robust, logreg, rf, irv, graphconv, xgb,' + \
         ' dag, weave, tf_regression, tf_regression_ft, rf_regression, ' + \
         'graphconvreg, xgb_regression, dtnn, dag_regression, weave_regression')
parser.add_argument(
    '-d',
    action='append',
    dest='dataset_args',
    default=[],
    help='Choice of dataset: bace_c, bace_r, bbbp, chembl, clearance, ' +
    'clintox, delaney, hiv, hopv, kaggle, lipo, muv, nci, pcba, ' +
    'pdbbind, ppb, qm7, qm7b, qm8, qm9, sampl, sider, tox21, toxcast')
parser.add_argument(
    '--seed',
    action='append',
    dest='seed_args',
    default=[],
    help='Choice of random seed')

args = parser.parse_args()
#Datasets and models used in the benchmark test
splitters = args.splitter_args
models = args.model_args
datasets = args.dataset_args

if len(args.seed_args) > 0:
  seed = int(args.seed_args[0])
else:
  seed = 123

out_path = '.'
for dataset in datasets:
  for split in splitters:
    for model in models:

      hyper_parameters = None
      # Uncomment the two lines below if hyper_parameters are provided
      #with open(os.path.join(out_path, dataset + model + '.pkl'), 'r') as f:
      #  hyper_parameters = pickle.load(f)

      if dataset in [
          'bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'sider', 'tox21',
          'toxcast'
      ]:
        mode = 'classification'
        metric = [dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)]
      elif dataset in [
          'bace_r', 'chembl', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo',
          'nci', 'pdbbind', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl'
      ]:
        mode = 'regression'
        metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)]

      pair = (dataset, model)
      if pair in CheckFeaturizer:
        featurizer = CheckFeaturizer[pair][0]
        n_features = CheckFeaturizer[pair][1]
      else:
        supported_combinations = [
            key for key in CheckFeaturizer.keys() if pair[0] == key[0]
        ]
        supported_models = [k[1] for k in supported_combinations]
        raise ValueError(
            "Model %s not supported for %s dataset. Please choose from the following:\n%s"
            % (pair[1], pair[0], "  ".join(supported_models)))

      tasks, all_dataset, transformers = load_dataset(
          dataset, featurizer, split='index')
      all_dataset = dc.data.DiskDataset.merge(all_dataset)
      for frac_train in frac_trains:
        splitters = {
            'index': dc.splits.IndexSplitter(),
            'random': dc.splits.RandomSplitter(),
            'scaffold': dc.splits.ScaffoldSplitter(),
            'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
        }
        splitter = splitters[split]
        np.random.seed(seed)
        train, valid, test = splitter.train_valid_test_split(
            all_dataset,
            frac_train=frac_train,
            frac_valid=1 - frac_train,
            frac_test=0.)
        test = valid
        if mode == 'classification':
          train_score, valid_score, test_score = benchmark_classification(
              train,
              valid,
              test,
              tasks,
              transformers,
              n_features,
              metric,
              model,
              test=False,
              hyper_parameters=hyper_parameters,
              seed=seed)
        elif mode == 'regression':
          train_score, valid_score, test_score = benchmark_regression(
              train,
              valid,
              test,
              tasks,
              transformers,
              n_features,
              metric,
              model,
              test=False,
              hyper_parameters=hyper_parameters,
              seed=seed)
        with open(os.path.join(out_path, 'results_frac_train_curve.csv'),
                  'a') as f:
          writer = csv.writer(f)
          model_name = list(train_score.keys())[0]
          for i in train_score[model_name]:
            output_line = [
                dataset,
                str(split), mode, model_name, i, 'train',
                train_score[model_name][i], 'valid', valid_score[model_name][i]
            ]
            output_line.extend(['frac_train', frac_train])
            writer.writerow(output_line)
