# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 22:41:34 2016

@author: Zhenqin Wu
"""
import numpy as np
from benchmark import benchmark_loading_datasets
import os
import time
import sys
from scipy.stats import truncnorm

# main layer
layer_sizes_0 = [1200]
weight_init_stddevs_0 = [0.02]
bias_init_consts_0 = [1.]
dropouts_0 = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]

#bypass layer
bypass_layer_sizes_0 = truncnorm(-1,1,loc=200,scale=150)
bypass_weight_init_stddevs_0 = [.02]
bypass_bias_init_consts_0 = [1.]
bypass_dropouts_0 = [0.4,0.45,0.5,0.55,0.6,0.65,0.7]

#penalty
penalty_0 = truncnorm(-1,1,loc=0.3,scale=0.2)
penalty_type_0 = ['l2']

#general figure
batch_size_0 = [50]
nb_epoch_0 = [12]

#learning rate
learning_rate_0 = truncnorm(-1,1,loc=-3.2,scale=1.2)

#for graph-conv and random forest
n_filters_0 = [64,96,128]
n_fully_connected_nodes_0 = [100,120,140,160,200,240,300]
n_estimators_0 = [500]
seed = None

out_path='.'
base_dir_o="/tmp/benchmark_test_"+time.strftime("%Y_%m_%d", time.localtime())

dname = sys.argv[1]
model = sys.argv[2]

parameters_printed = {'tf':['layer_sizes', 'weight_init_stddevs',
                            'bias_init_consts', 'dropouts', 'penalty',
                            'penalty_type', 'batch_size', 'nb_epoch', 
                            'learning_rate'],
                      'tf_robust':['layer_sizes', 'weight_init_stddevs',
                                   'bias_init_consts', 'dropouts',
                                   'bypass_layer_sizes',
                                   'bypass_weight_init_stddevs',
                                   'bypass_bias_init_consts',
                                   'bypass_dropouts', 'penalty', 
                                   'penalty_type', 'batch_size', 
                                   'nb_epoch', 'learning_rate'],
                      'logreg':['penalty', 'penalty_type', 'batch_size', 
                                'nb_epoch', 'learning_rate'],
                      'graphconv':['batch_size', 'nb_epoch', 'learning_rate', 
                                   'n_filters', 'n_fully_connected_nodes'],
                      'rf':['n_estimators']}
hps = {}
hps[model] = []
for i in range(int(sys.argv[3])):
  layer_sizes = layer_sizes_0
  weight_init_stddevs = weight_init_stddevs_0
  bias_init_consts = bias_init_consts_0
  dropouts = [np.random.choice(dropouts_0)]
  
  bypass_layer_sizes = [int(bypass_layer_sizes_0.rvs())]
  bypass_weight_init_stddevs = bypass_weight_init_stddevs_0
  bypass_bias_init_consts = bypass_bias_init_consts_0
  bypass_dropouts = [np.random.choice(bypass_dropouts_0)]

  penalty = penalty_0.rvs()
  penalty_type = np.random.choice(penalty_type_0)

  batch_size = np.random.choice(batch_size_0)
  nb_epoch = np.random.choice(nb_epoch_0)

  learning_rate = 10**(learning_rate_0.rvs())

  n_filters = np.random.choice(n_filters_0)
  n_fully_connected_nodes = np.random.choice(n_fully_connected_nodes_0)
  n_estimators = np.random.choice(n_estimators_0)
  
  hps[model].append({'layer_sizes': layer_sizes,
      'weight_init_stddevs': weight_init_stddevs,
      'bias_init_consts': bias_init_consts,
      'dropouts': dropouts, 'bypass_layer_sizes': bypass_layer_sizes, 
      'bypass_weight_init_stddevs': bypass_weight_init_stddevs, 
      'bypass_bias_init_consts': bypass_bias_init_consts, 
      'bypass_dropouts': bypass_dropouts, 
      'penalty': penalty, 'penalty_type': penalty_type, 
      'batch_size': batch_size, 'nb_epoch': nb_epoch,
      'learning_rate': learning_rate, 'n_filters': n_filters,
      'n_fully_connected_nodes': n_fully_connected_nodes, 
      'n_estimators': n_estimators, 'seed': seed})
  
  with open(os.path.join(out_path,'hps.csv'),'a') as f:        
    f.write('\n'+str(i)+','+dname+',')
    for item in hps[model][i]:
      if item in parameters_printed[model]:
        f.write(item+',')
        f.write(str(hps[model][i][item])+',')

benchmark_loading_datasets(base_dir_o, hps, dataset=dname,
                           model=model, reload=True,
                           verbosity='high', out_path=out_path)



    
             
