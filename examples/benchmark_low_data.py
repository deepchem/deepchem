#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:48:05 2016

@author: Michael Wu

Low data benchmark test
Giving performances of: Siamese, attention-based embedding, residual embedding
                    
on datasets: muv, sider, tox21

time estimation listed in README file

"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
import argparse

np.random.seed(123)

parser = argparse.ArgumentParser(
    description='Deepchem benchmark: ' +
    'giving performances of different learning models on datasets')
parser.add_argument(
    '-m',
    action='append',
    dest='model_args',
    default=[],
    help='Choice of model: siamese, attn, res')
parser.add_argument(
    '-d',
    action='append',
    dest='dataset_args',
    default=[],
    help='Choice of dataset: tox21, sider, muv')
parser.add_argument(
    '--cv',
    action='store_true',
    dest='cross_valid',
    default=False,
    help='whether to implement cross validation')

args = parser.parse_args()
#Datasets and models used in the benchmark test
models = args.model_args
datasets = args.dataset_args
cross_valid = args.cross_valid

if len(models) == 0:
  models = ['siamese', 'attn', 'res']
if len(datasets) == 0:
  datasets = ['tox21', 'sider', 'muv']

for dataset in datasets:
  for model in models:
    dc.molnet.run_benchmark_low_data(
        [dataset], str(model), cross_valid=cross_valid)
