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

Hyperparameters and all benchmark scripts for MoleculeNet are available at:
http://deepchem.io.s3-website-us-west-1.amazonaws.com/trained_models/Hyperparameter_MoleculeNetv3.tar.gz

"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import deepchem as dc
import argparse

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
    '-t',
    action='store_true',
    dest='test',
    default=False,
    help='Evalute performance on test set')
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
test = args.test
if len(args.seed_args) > 0:
  seed = int(args.seed_args[0])
else:
  seed = 123

if len(splitters) == 0:
  splitters = ['index', 'random', 'scaffold']
if len(models) == 0:
  models = [
      'tf', 'tf_robust', 'logreg', 'graphconv', 'irv', 'tf_regression',
      'tf_regression_ft', 'graphconvreg', 'weave', 'weave_regression', 'dtnn'
  ]
  #irv, rf, rf_regression should be assigned manually
if len(datasets) == 0:
  datasets = [
      'bace_c', 'bace_r', 'bbbp', 'clearance', 'clintox', 'delaney', 'hiv',
      'hopv', 'lipo', 'muv', 'pdbbind', 'ppb', 'qm7b', 'qm8', 'qm9', 'sampl',
      'sider', 'tox21', 'toxcast'
  ]

for dataset in datasets:
  for split in splitters:
    for model in models:
      np.random.seed(seed)
      dc.molnet.run_benchmark(
          [dataset], str(model), split=split, test=test, seed=seed)
