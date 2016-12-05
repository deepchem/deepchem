"""
@author: Bharath Ramsundar 
"""
import os
import time
import sys
import pandas as pd
import numpy as np
import tempfile
from scipy.stats import truncnorm
from benchmark import benchmark_loading_datasets

np.random.seed(123)
base_dir = tempfile.mkdtemp()
out_path='.'
models = ['tf', 'tf_robust', 'logreg', 'graphconv']
#datasets = ['muv', 'nci', 'tox21', 'sider', 'toxcast']
datasets = ['tox21', 'muv']

hps = {}
hps['tf'] = [{'dropouts': [0.25], 'learning_rate': 0.001,
              'layer_sizes': [1000], 'batch_size': 50, 'nb_epoch': 10}]

hps['tf_robust'] = [{'dropouts': [0.5], 'bypass_dropouts': [0.5],
                     'learning_rate': 0.001,
                     'layer_sizes': [500], 'bypass_layer_sizes': [100],
                     'batch_size': 50, 'nb_epoch': 10}]
              
hps['logreg'] = [{'learning_rate': 0.001, 'penalty': 0.05, 
                  'penalty_type': 'l1', 'batch_size': 50, 'nb_epoch': 10}]
              
hps['graphconv'] = [{'learning_rate': 0.001, 'n_filters': 64,
                     'n_fully_connected_nodes': 128, 'batch_size': 50,
                     'nb_epoch': 10}]

for model in models:
  for dataset in datasets:
    print("Benchmarking %s on dataset %s" % (model, dataset))
    benchmark_loading_datasets(base_dir, hps, dataset=dataset,
                               model=model, reload=True,
                               verbosity='high', out_path=out_path)
