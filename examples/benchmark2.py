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
    Graph convolution(graphconv), xgboost(xgb),
    Directed acyclic graph(dag), Weave(weave)
on datasets: bace_c, bbbp, clintox, hiv, muv, pcba, sider, tox21, toxcast

Giving regression performances of:
    MultitaskDNN(tf_regression),
    Fit Transformer MultitaskDNN(tf_regression_ft),
    Random forest(rf_regression),
    Graph convolution regression(graphconvreg),
    xgboost(xgb_regression), Deep tensor neural net(dtnn),
    Directed acyclic graph(dag_regression),
    Weave(weave_regression)
on datasets: bace_r, chembl, clearance, delaney(ESOL), hopv, kaggle, lipo,
             nci, pdbbind, ppb, qm7, qm7b, qm8, qm9, sampl(FreeSolv)


time estimation listed in README file

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
from deepchem.molnet.check_availability import CheckFeaturizer, CheckSplit
from deepchem.molnet.preset_hyper_parameters import hps

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
splits = args.splitter_args
models = args.model_args
datasets = args.dataset_args
if len(args.seed_args) > 0:
  seed = int(args.seed_args[0])
else:
  seed = 123

if len(splits) == 0:
  splits = ['random']
if len(models) == 0:
  models = [
      'tf', 'tf_robust', 'logreg', 'graphconv', 'irv', 'tf_regression',
      'tf_regression_ft', 'graphconvreg', 'weave', 'weave_regression', 'dtnn'
  ]
  #irv, rf, rf_regression should be assigned manually
if len(datasets) == 0:
  datasets = [
      'clintox', 'delaney', 'lipo', 'qm7b', 'qm8', 'sampl',
      'sider', 'tox21', 'toxcast', 'muv'
  ]

metrics = {
    'qm7': [[dc.metrics.Metric(dc.metrics.mean_absolute_error, np.mean, mode='regression')], False],
    'qm7b': [[dc.metrics.Metric(dc.metrics.mean_absolute_error, np.mean, mode='regression')], False],
    'qm8': [[dc.metrics.Metric(dc.metrics.mean_absolute_error, np.mean, mode='regression')], False],
    'qm9': [[dc.metrics.Metric(dc.metrics.mean_absolute_error, np.mean, mode='regression')], False],
    'delaney': [[dc.metrics.Metric(dc.metrics.rms_score, np.mean, mode='regression')], False],
    'sampl': [[dc.metrics.Metric(dc.metrics.rms_score, np.mean, mode='regression')], False],
    'lipo': [[dc.metrics.Metric(dc.metrics.rms_score, np.mean, mode='regression')], False],
    'pdbbind': [[dc.metrics.Metric(dc.metrics.rms_score, np.mean, mode='regression')], False],
    'pcba': [[dc.metrics.Metric(dc.metrics.prc_auc_score, np.mean, mode='classification')], True],
    'muv': [[dc.metrics.Metric(dc.metrics.prc_auc_score, np.mean, mode='classification')], True],
    'hiv': [[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')], True],
    'tox21': [[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')], True],
    'toxcast': [[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')], True],
    'sider': [[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')], True],
    'clintox': [[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')], True],
    'bace_c': [[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')], True],
    'bbbp': [[dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')], True]
    }
out_path = '.'
for dataset in datasets:
  for split in splits:
    for model in models:
      with open(os.path.join(out_path, dataset + model + '.pkl'), 'r') as f:
        hyper_parameters = pickle.load(f)
      #hyper_parameters = None

      metric = metrics[dataset][0]
      if dataset in [
        'bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'sider', 'tox21',
        'toxcast'
        ]:
        mode = 'classification'
      elif dataset in [
        'bace_r', 'chembl', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo',
        'nci', 'pdbbind', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl'
      ]:
        mode = 'regression'

      pair = (dataset, model)
      if pair in CheckFeaturizer:
        featurizer = CheckFeaturizer[pair][0]
        n_features = CheckFeaturizer[pair][1]

      loading_functions = {
        'bace_c': dc.molnet.load_bace_classification,
        'bace_r': dc.molnet.load_bace_regression,
        'bbbp': dc.molnet.load_bbbp,
        'chembl': dc.molnet.load_chembl,
        'clearance': dc.molnet.load_clearance,
        'clintox': dc.molnet.load_clintox,
        'delaney': dc.molnet.load_delaney,
        'hiv': dc.molnet.load_hiv,
        'hopv': dc.molnet.load_hopv,
        'kaggle': dc.molnet.load_kaggle,
        'lipo': dc.molnet.load_lipo,
        'muv': dc.molnet.load_muv,
        'nci': dc.molnet.load_nci,
        'pcba': dc.molnet.load_pcba,
        'pdbbind': dc.molnet.load_pdbbind_grid,
        'ppb': dc.molnet.load_ppb,
        'qm7': dc.molnet.load_qm7_from_mat,
        'qm7b': dc.molnet.load_qm7b_from_mat,
        'qm8': dc.molnet.load_qm8,
        'qm9': dc.molnet.load_qm9,
        'sampl': dc.molnet.load_sampl,
        'sider': dc.molnet.load_sider,
        'tox21': dc.molnet.load_tox21,
        'toxcast': dc.molnet.load_toxcast
        }

      tasks, all_dataset, transformers = loading_functions[dataset](
          featurizer=featurizer, reload=reload, split='index')
      all_dataset = dc.data.DiskDataset.merge(all_dataset)
      for seed in [122, 123, 124]:
          splitters = {
            'random': dc.splits.RandomSplitter(),
            'scaffold': dc.splits.ScaffoldSplitter(),
            'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
          }
          splitter = splitters[split]
          np.random.seed(seed)
          train, valid, test = splitter.train_valid_test_split(all_dataset,
                                                               frac_train=0.8,
                                                               frac_valid=0.1,
                                                               frac_test=0.1)
          if mode == 'classification':
            train_score, valid_score, test_score = benchmark_classification(
                train, valid, test, tasks, transformers, n_features, metric,
                model, test=True, hyper_parameters=hyper_parameters, seed=seed)
          elif mode == 'regression':
            train_score, valid_score, test_score = benchmark_regression(
                train, valid, test, tasks, transformers, n_features, metric,
                model, test=True, hyper_parameters=hyper_parameters, seed=seed)
          with open(os.path.join(out_path, 'final/results.csv'), 'a') as f:
            writer = csv.writer(f)
            model_name = list(train_score.keys())[0]
            for i in train_score[model_name]:
              output_line = [
                dataset, str(split), mode, model_name, i, 'train',
                str(train_score[model_name]), 'valid', str(valid_score[model_name]),
                'test', str(test_score[model_name])
              ]
              writer.writerow(output_line)
