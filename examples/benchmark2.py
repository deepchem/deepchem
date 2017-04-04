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
    Directed acyclic graph(dag) 
on datasets: bace_c, bbbp, clintox, hiv, muv, pcba, sider, tox21, toxcast  

Giving regression performances of:
    MultitaskDNN(tf_regression),
    Fit Transformer MultitaskDNN(tf_regression_ft),
    Random forest(rf_regression),
    Graph convolution regression(graphconvreg),
    xgboost(xgb_regression), Deep tensor neural net(dtnn),
    Directed acyclic graph(dag_regression)
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

np.random.seed(123)

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
    help='Choice of model: tf, tf_robust, logreg, rf, irv, graphconv, ' +
    'tf_regression, tf_regression_ft, rf_regression, graphconvreg')
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
      'tf_regression_ft', 'graphconvreg'
  ]
  #irv, rf, rf_regression should be assigned manually
if len(datasets) == 0:
  datasets = [
      'bace_c', 'bace_r', 'bbbp', 'clearance', 'clintox', 'delaney', 'hiv',
      'hopv', 'lipo', 'muv', 'pcba', 'pdbbind', 'ppb', 'qm7b', 'qm8', 'qm9',
      'sampl', 'sider', 'tox21', 'toxcast'
  ]

for split in splitters:
  for dataset in datasets:
    for model in models:
      dc.molnet.run_benchmark([dataset], str(model), split=split, test=test)
