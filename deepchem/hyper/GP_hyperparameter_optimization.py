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
import deepchem
from deepchem.molnet.preset_hyper_parameters import hps
import csv
import copy
from deepchem.utils.dependencies import pyGPGO_covfunc, pyGPGO_acquisition, \
    pyGPGO_surrogates_GaussianProcess, pyGPGO_GPGO


def GaussianProcessHyperparamOpt(dataset='tox21',
                                 model='tf',
                                 split='random',
                                 max_iter=20,
                                 search_range=4,
                                 hp_invalid_list=[
                                     'seed', 'nb_epoch', 'penalty_type',
                                     'dropouts', 'bypass_dropouts',
                                     'n_pair_feat'
                                 ]):
  hyper_parameters = copy.deepcopy(hps[model])
  # Extract parameters
  hp_list = hyper_parameters.keys()
  for hp in hp_invalid_list:
    if hp in hp_list:
      hp_list.remove(hp)

  hp_list_class = [hyper_parameters[hp].__class__ for hp in hp_list]
  assert set(hp_list_class) <= set([list, int, float])
  # Float or int hyper parameters(ex. batch_size, learning_rate)
  hp_list1 = [
      hp_list[i] for i in range(len(hp_list)) if not hp_list_class[i] is list
  ]
  # List of float or int hyper parameters(ex. layer_sizes)
  hp_list2 = [(hp_list[i], len(hyper_parameters[hp_list[i]]))
              for i in range(len(hp_list)) if hp_list_class[i] is list]

  # Number of parameters
  n_param = len(hp_list1) + sum([hp[1] for hp in hp_list2])
  # Range of optimization
  param_range = []
  for hp in hp_list1:
    if hyper_parameters[hp].__class__ is int:
      param_range.append((('int'), [
          hyper_parameters[hp] / search_range,
          hyper_parameters[hp] * search_range
      ]))
    else:
      param_range.append((('cont'), [
          hyper_parameters[hp] / search_range,
          hyper_parameters[hp] * search_range
      ]))
  for hp in hp_list2:
    if hyper_parameters[hp[0]][0].__class__ is int:
      param_range.extend([(('int'), [
          hyper_parameters[hp[0]][i] / search_range,
          hyper_parameters[hp[0]][i] * search_range
      ]) for i in range(hp[1])])
    else:
      param_range.extend([(('cont'), [
          hyper_parameters[hp[0]][i] / search_range,
          hyper_parameters[hp[0]][i] * search_range
      ]) for i in range(hp[1])])

  # Dummy names
  param_name = ['l' + format(i, '02d') for i in range(20)]
  param = dict(zip(param_name[:n_param], param_range))

  def f(l00=0,
        l01=0,
        l02=0,
        l03=0,
        l04=0,
        l05=0,
        l06=0,
        l07=0,
        l08=0,
        l09=0,
        l10=0,
        l11=0,
        l12=0,
        l13=0,
        l14=0,
        l15=0,
        l16=0,
        l17=0,
        l18=0,
        l19=0):
    args = locals()
    # Input hyper parameters
    i = 0
    for hp in hp_list1:
      hyper_parameters[hp] = float(args[param_name[i]])
      if param_range[i][0] == 'int':
        hyper_parameters[hp] = int(hyper_parameters[hp])
      i = i + 1
    for hp in hp_list2:
      hyper_parameters[hp[0]] = [
          float(args[param_name[j]]) for j in range(i, i + hp[1])
      ]
      if param_range[i][0] == 'int':
        hyper_parameters[hp[0]] = map(int, hyper_parameters[hp[0]])
      i = i + hp[1]

    print(hyper_parameters)
    # Run benchmark
    deepchem.molnet.run_benchmark(
        [dataset],
        str(model),
        split=str(split),
        hyper_parameters=hyper_parameters,
        out_path='/tmp')
    # Read results
    with open('/tmp/results.csv', 'r') as f:
      reader = csv.reader(f)
      # Return valid set performances
      return float(list(reader)[-1][8])

  cov = pyGPGO_covfunc.matern32()
  gp = pyGPGO_surrogates_GaussianProcess.GaussianProcess(cov)
  acq = pyGPGO_acquisition.Acquisition(mode='ExpectedImprovement')
  gpgo = pyGPGO_GPGO.GPGO(gp, acq, f, param)
  gpgo.run(max_iter=max_iter)

  hp_opt, valid_performance_opt = gpgo.getResult()

  # Readout best hyper parameters
  i = 0
  for hp in hp_list1:
    hyper_parameters[hp] = float(hp_opt[param_name[i]])
    if param_range[i][0] == 'int':
      hyper_parameters[hp] = int(hyper_parameters[hp])
    i = i + 1
  for hp in hp_list2:
    hyper_parameters[hp[0]] = [
        float(hp_opt[param_name[j]]) for j in range(i, i + hp[1])
    ]
    if param_range[i][0] == 'int':
      hyper_parameters[hp[0]] = map(int, hyper_parameters[hp[0]])
    i = i + hp[1]

  return hyper_parameters, valid_performance_opt
