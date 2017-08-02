#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:30:59 2017

@author: zqwu
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from deepchem.molnet.preset_hyper_parameters import hps
import csv

from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

model = 'graphconvreg'
dataset = 'delaney'
split = 'random'
hyper_parameters = hps[model]
# Extract parameters
hp_list = hyper_parameters.keys()
hp_list.remove('seed')
hp_list.remove('nb_epoch')
hp_list_class = [hyper_parameters[hp].__class__ for hp in hp_list]

int_cont = []
for hp in hp_list_class:
  if hp is int:
    int_cont.append('int')
  else:
    int_cont.append('cont')

# Range of optimization
param_range = [(int_cont[i], [
    hyper_parameters[hp_list[i]] / 4, hyper_parameters[hp_list[i]] * 4
]) for i in range(len(int_cont))]
# Number of parameters
n_param = len(param_range)
# Dummy names
param_name = ['l' + str(i) for i in range(10)]
param = dict(zip(param_name[:n_param], param_range))


def f(l0=0, l1=0, l2=0, l3=0, l4=0, l5=0, l6=0, l7=0, l8=0, l9=0):
  args = locals()
  keys = args.keys()
  keys.sort()
  # Input hyper parameters
  for i, hp in enumerate(hp_list):
    hyper_parameters[hp] = args[keys[i]]
    if int_cont[i] == 'int':
      hyper_parameters[hp] = int(hyper_parameters[hp])
  print(hyper_parameters)
  # Run benchmark
  dc.molnet.run_benchmark(
      [dataset],
      str(model),
      split=str(split),
      hyper_parameters=hyper_parameters,
      out_path='/tmp')
  # Read results
  with open('/tmp/results.csv', 'r') as f:
    reader = csv.reader(f)
    return float(list(reader)[-1][8])


cov = matern32()
gp = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')
gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=10)
gpgo.getResult()
