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

layer_sizes = [1200]
dropouts = [0.25,0.30,0.35,0.40,0.45,0.50]
penalty_distri = truncnorm(-1,1,loc=0.05,scale=0.04)
penalty_type = ['l1']
batch_size = [100]
nb_epoch = [50]
learning_rate_distri = truncnorm(-0.3,0.3,loc=-3,scale=0.5)

n_filters = [64,96,128]
n_fully_connected_nodes = [100,120,140,160,200,240,300]
n_estimators = [500]

out_path='/home/zqwu/deepchem/examples'
base_dir_o="/tmp/benchmark_test_"+time.strftime("%Y_%m_%d", time.localtime())
dname = sys.argv[1]
model = sys.argv[2]

parameters_printed = {'tf':['dropouts','learning_rate','layer_sizes','penalty',
                            'batch_size','nb_epoch'],
                      'logreg':['learning_rate','penalty','penalty_type',
                                'batch_size','nb_epoch'],
                      'graphconv':['learning_rate','n_filters','n_fully_connected_nodes',
                                   'batch_size','nb_epoch'],
                      'rf':['n_estimators']}
hps = {}
hps[model] = []
for i in range(int(sys.argv[3])):
  ls = np.random.choice(layer_sizes)
  dp = np.random.choice(dropouts)
  pn = penalty_distri.rvs()
  pt = np.random.choice(penalty_type)
  bs = np.random.choice(batch_size)

  lr = 10.**(learning_rate_distri.rvs())
  ne = np.min([np.ceil(0.02/lr),nb_epoch[0]])

  nf = np.random.choice(n_filters)
  nn = np.random.choice(n_fully_connected_nodes)
  nest = np.random.choice(n_estimators)



  hps[model].append({'dropouts':[dp],'learning_rate':lr,'layer_sizes':[int(ls)],
                'penalty':pn, 'penalty_type':pt, 'batch_size':int(bs),'nb_epoch':int(ne),
		'n_filters': 64, 'n_fully_connected_nodes':128, 'n_estimators':500})
  with open(os.path.join(out_path,'hps.csv'),'a') as f:        
    f.write('\n\n'+str(i))
    for item in hps[model][i]:
      if item in parameters_printed[model]:
        f.write('\n'+item+',')
        f.write(str(hps[model][i][item]))

benchmark_loading_datasets(base_dir_o, hps, n_features = 1024, 
                           dataset_name=dname,model=model,reload = True,
                           verbosity='high', out_path=out_path)



    
             
