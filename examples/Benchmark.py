#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:53:27 2016

@author: Michael Wu

Benchmark test
Giving performances of RF(scikit) and MultitaskDNN(TF)
on datasets: muv, nci, pcba, tox21

time estimation(on a nvidia tesla K20 GPU):
tox21 - dataloading: 30s
      - tf: 40s
muv   - dataloading: 400s
      - tf: 250s
pcba  - dataloading: 30min
      - tf: 2h
sider - dataloading: 10s
      - tf: 60s
toxcast dataloading: 70s
	tf: 70s
(will include more)
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import os
import numpy as np
import shutil
import time
import deepchem as dc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from muv.muv_datasets import load_muv
from nci.nci_datasets import load_nci
from pcba.pcba_datasets import load_pcba
from tox21.tox21_datasets import load_tox21
from toxcast.toxcast_datasets import load_toxcast
from sider.sider_datasets import load_sider

def benchmark_loading_datasets(base_dir_o, hyper_parameters, n_features = 1024, 
                               dataset_name='all',model='all',reload = True,
                               verbosity='high',out_path='/home/zqwu/deepchem/examples'):
  """
  Loading dataset for benchmark test
  
  Parameters
  ----------
  base_dir_o, string
      path of working folder, will be combined with '/dataset_name'
  
  hyper_parameters, dict of list
      hyper parameters including dropout rate, learning rate, etc.
 
  n_features, integer, optional (default=1024)
      number of features, or length of binary fingerprints
  
  dataset_name, string, optional (default='all')
      choice of which dataset to use, 'all' = computing all the datasets
      
  model string, optional (default='all')
      choice of which model to use, 'all' = running all models on the dataset
  
  out_path, string, optional(default='/tmp')
      path of result file
      
  """
  assert dataset_name in ['all', 'muv', 'nci', 'pcba', 'tox21','sider',
                          'toxcast']
  
  if dataset_name == 'all':
    #currently not including the nci dataset
    dataset_name = ['tox21','muv','pcba','sider','toxcast']
  else:
    dataset_name = [dataset_name]
  
  loading_functions = {'tox21':load_tox21, 'muv':load_muv,
                       'pcba':load_pcba, 'nci':load_nci,
                       'sider':load_sider, 'toxcast':load_toxcast}
  
  for dname in dataset_name:
    print('-------------------------------------')
    print('Benchmark test on dataset: '+dname)
    print('-------------------------------------')
    base_dir = os.path.join(base_dir_o, dname)
    
    time_start = time.time()
    #loading datasets     
    tasks,datasets,transformers = loading_functions[dname]()
    train_dataset, valid_dataset, test_dataset = datasets
    time_finish_loading = time.time()
    #time_finish_loading-time_start is the time(s) used for dataset loading
    

    #running model
    train_score,valid_score = benchmark_train_and_valid(base_dir,train_dataset,
                                                        valid_dataset, tasks,
                                                        transformers,
                                                        hyper_parameters,
                                                        n_features=n_features,
                                                        model = model,
                                                        verbosity = verbosity)
    time_finish_running = time.time()
    #time_finish_running-time_finish_loading is the time(s) used for fitting and evaluating
        
    with open(os.path.join(out_path,'results.csv'),'a') as f:
      f.write ('\n'+dname+',train')
      for i in train_score:
        f.write(','+i+','+str(train_score[i]['mean-roc_auc_score']))
      f.write('\n'+dname+',valid')
      for i in valid_score:
        f.write(','+i+','+str(valid_score[i]['mean-roc_auc_score'])) 
      #output timing data: running time include all the model
      f.write('\n'+dname+',time_for_loading,,'+
              str(time_finish_loading-time_start)+'seconds')
      f.write('\n'+dname+',time_for_running,,'+
              str(time_finish_running-time_finish_loading)+'seconds')
    
    #clear workspace         
    del tasks,datasets,transformers
    del train_dataset,valid_dataset, test_dataset
    del time_start,time_finish_loading,time_finish_running

  return None

def benchmark_train_and_valid(base_dir,train_dataset,valid_dataset,tasks,
                              transformers, hyper_parameters,
                              n_features = 1024,model = 'all',
                              verbosity = 'high'):
  """
  Calculate performance of different models on the specific dataset & tasks
  
  Parameters
  ----------
  base_dir, string
      path of working folder
      
  train_dataset, dataset struct
      loaded dataset using load_* or splitter function
      
  valid_dataset, dataset struct
      loaded dataset using load_* or splitter function
  
  tasks, list of string
      list of targets(tasks, datasets)
  
  transformers, BalancingTransformer struct
      loaded properties of dataset from load_* function
  
  hyper_parameters, dict of list
      hyper parameters including dropout rate, learning rate, etc.
 
  n_features, integer, optional (default=1024)
      number of features, or length of binary fingerprints
  
  model, string, optional (default='all')
      choice of which model to use, 'all' = running all models on the dataset
      
  """
  train_scores = {}
  valid_scores = {}
  
  # Initialize metrics
  classification_metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean,
                                            verbosity=verbosity,
                                            mode="classification")
  
  assert model in ['all', 'tf', 'rf']

  if model == 'all' or model == 'tf':
    # Initialize model folder
    model_dir_tf = os.path.join(base_dir, "model_tf")
    
    dropouts = hyper_parameters['tf'][0]
    learning_rate = hyper_parameters['tf'][1]
    weight_init_stddevs = hyper_parameters['tf'][2]
    batch_size = hyper_parameters['tf'][3]
    # Building tensorflow MultiTaskDNN model
    tensorflow_model = dc.models.TensorflowMultiTaskClassifier(
        len(tasks), n_features, dropouts=[dropouts],
        learning_rate=learning_rate, weight_init_stddevs=[weight_init_stddevs],
        batch_size=batch_size, verbosity=verbosity)
    model_tf = dc.models.TensorflowModel(tensorflow_model)
 
    print('-------------------------------------')
    print('Start fitting by tensorflow')
    model_tf.fit(train_dataset)

    train_scores['tensorflow'] = model_tf.evaluate(train_dataset,
                                    [classification_metric],transformers)

    valid_scores['tensorflow'] = model_tf.evaluate(valid_dataset,
                                    [classification_metric],transformers)

  
  if model == 'all' or model == 'rf':
    # Initialize model folder
    model_dir_rf = os.path.join(base_dir, "model_rf")
    
    # Building scikit random forest model
    def model_builder(model_dir_rf):
      sklearn_model = RandomForestClassifier(
        class_weight="balanced", n_estimators=500,n_jobs=-1)
      return dc.models.sklearn_models.SklearnModel(sklearn_model, model_dir_rf)
    model_rf = dc.models.multitask.SingletaskToMultitask(
		tasks, model_builder, model_dir_rf)
    
    print('-------------------------------------')
    print('Start fitting by random forest')
    model_rf.fit(train_dataset)
    train_scores['random_forest'] = model_rf.evaluate(train_dataset,
                                    [classification_metric],transformers)

    valid_scores['random_forest'] = model_rf.evaluate(valid_dataset,
                                    [classification_metric],transformers)

  return train_scores, valid_scores

if __name__ == '__main__':
  # Global variables
  np.random.seed(123)
  verbosity = 'high'
  
  #Working folder initialization
  base_dir_o="/tmp/benchmark_test_"+time.strftime("%Y_%m_%d", time.localtime())
  if os.path.exists(base_dir_o):
    shutil.rmtree(base_dir_o)
  os.makedirs(base_dir_o)
  
  #Datasets and models used in the benchmark test, all=all the datasets(models)
  dataset_name = sys.argv[1]
  model = sys.argv[2]

  #input hyperparameters
  #tf: dropouts, learning rate, weight initial stddev, batch_size
  hyper_parameters = {'tf':[0.25, 0.0003, 0.1, 50]}

  benchmark_loading_datasets(base_dir_o,hyper_parameters,n_features = 1024,
                             dataset_name = dataset_name, model = model,
                             reload = reload, verbosity = verbosity)
