#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:53:27 2016

@author: Michael Wu

Benchmark test
Giving performances of RF(scikit) and MultitaskDNN(Keras & TF)
on datasets: muv, nci, pcba, tox21

"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import os
import numpy as np
import shutil
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from deepchem.datasets import Dataset
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.utils.evaluate import Evaluator
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.keras_models import KerasModel
from deepchem.models.multitask import SingletaskToMultitask
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.splits import RandomSplitter

from muv.muv_datasets import load_muv
from nci.nci_datasets import load_nci
from pcba.pcba_datasets import load_pcba
from tox21.tox21_datasets import load_tox21

def benchmarkLoadingDatasets(base_dir_o, n_features = 1024, datasetName = 'all',
                             model = 'all', reload = True, verbosity = 'high'):
  
  assert datasetName in ['all', 'muv', 'nci', 'pcba', 'tox21']
  
  if datasetName == 'all':
    datasetName = ['muv','pcba','tox21']
  else:
    datasetName = [datasetName]
      
  if 'tox21' in datasetName:
    print('-------------------------------------')
    print('Benchmark test on dataset: tox21')
    print('-------------------------------------')
    base_dir = os.path.join(base_dir_o, "tox21")
    time1 = time.time()
    #loading datasets for tox21
    tasks_tox21,datasets_tox21,transformers_tox21 = load_tox21(base_dir,
                                                               reload=reload)
    time2 = time.time()
    #time2-time1 is the time(s) used for dataset loading
    
    #dataset splitting, built-in method in load_tox21
    train_dataset, valid_dataset = datasets_tox21
    #running model
    tox21_train,tox21_valid = benchmarkTrainAndValid(base_dir,train_dataset,
                                                valid_dataset,
                                                tasks_tox21,transformers_tox21,
                                                n_features, model, verbosity)
    time3 = time.time()
    #time3-time2 is the time(s) used for fitting and evaluating
        
    with open('/home/zqwu/deepchem/examples/results.csv','a') as f:
      f.write ('\n'+'tox21,train')
      for i in tox21_train:
        f.write(','+i+','+str(tox21_train[i])) #output train score
      f.write('\n'+'tox21,valid')
      for i in tox21_valid:
        f.write(','+i+','+str(tox21_valid[i])) #output valid score
      #output timing data: running time include all the model
      f.write('\n'+'tox21,time_for_loading,'+str(time2-time1)+'seconds')
      f.write('\n'+'tox21,time_for_running,'+str(time3-time2)+'seconds')
    
    #clear workspace         
    del tasks_tox21,datasets_tox21,transformers_tox21
    del train_dataset,valid_dataset,time1,time2,time3

  if 'muv' in datasetName:
    print('-------------------------------------')
    print('Benchmark test on dataset: muv')
    print('-------------------------------------')
    base_dir = os.path.join(base_dir_o, "muv")
    time1 = time.time()
    #loading datasets for muv
    tasks_muv,datasets_muv,transformers_muv = load_muv(base_dir,reload=reload)
    time2 = time.time()    
    
    #dataset splitting, built-in method in load_tox21
    train_dataset, valid_dataset = datasets_muv
    #running model
    muv_train,muv_valid = benchmarkTrainAndValid(base_dir,train_dataset,
                                                 valid_dataset,
                                                 tasks_muv, transformers_muv,
                                                 n_features, model, verbosity)
    time3 = time.time()
    
    with open('/home/zqwu/deepchem/examples/results.csv','a') as f:
      f.write ('\n'+'muv,train')
      for i in muv_train:
        f.write(','+i+','+str(muv_train[i]))
      f.write('\n'+'muv,valid')
      for i in muv_valid:
        f.write(','+i+','+str(muv_valid[i]))
      f.write('\n'+'muv,time_for_loading,'+str(time2-time1)+'seconds')
      f.write('\n'+'muv,time_for_running,'+str(time3-time2)+'seconds')
      
    del tasks_muv, datasets_muv, transformers_muv
    del train_dataset,valid_dataset,time1,time2,time3

  if 'pcba' in datasetName:
    print('-------------------------------------')
    print('Benchmark test on dataset: pcba')
    print('-------------------------------------')
    base_dir = os.path.join(base_dir_o, "pcba")
    train_dir = os.path.join(base_dir, "train_dataset")
    valid_dir = os.path.join(base_dir, "valid_dataset")
    test_dir = os.path.join(base_dir, "test_dataset")

    time1 = time.time()
    #loading datasets for pcba
    tasks_pcba,datasets_pcba,transformers_pcba = load_pcba(base_dir,
                                                           reload=reload)
    time2 = time.time()
   
    #dataset splitting, RandomSplitter function
    print("About to perform train/valid/test split.")
    splitter = RandomSplitter(verbosity=verbosity)
    print("Performing new split.")
    train_dataset,valid_dataset,test_dataset = splitter.train_valid_test_split(
                                datasets_pcba, train_dir, valid_dir, test_dir)
    #running model
    pcba_train,pcba_valid = benchmarkTrainAndValid(base_dir,train_dataset,
                                                valid_dataset,
                                                tasks_pcba,transformers_pcba,
                                                n_features, model, verbosity)
    time3 = time.time()

    with open('/home/zqwu/deepchem/examples/results.csv','a') as f:
      f.write ('\n'+'pcba,train')
      for i in pcba_train:
        f.write(','+i+','+str(pcba_train[i]))
      f.write('\n'+'pcba,valid')
      for i in pcba_valid:
        f.write(','+i+','+str(pcba_valid[i]))
      f.write('\n'+'pcba,time_for_loading,'+str(time2-time1)+'seconds')
      f.write('\n'+'pcba,time_for_running,'+str(time3-time2)+'seconds')
     
    del tasks_pcba, datasets_pcba, transformers_pcba
    del train_dataset,valid_dataset,time1,time2,time3

  if 'nci' in datasetName:
    print('-------------------------------------')
    print('Benchmark test on dataset: nci')
    print('-------------------------------------')
    base_dir = os.path.join(base_dir_o,  "nci")
    train_dir = os.path.join(base_dir, "train_dataset")
    valid_dir = os.path.join(base_dir, "valid_dataset")
    test_dir = os.path.join(base_dir, "test_dataset")

    time1 = time.time()
    #loading datasets for nci
    tasks_nci,datasets_nci,transformers_nci = load_nci(base_dir, reload=reload)
    time2 = time.time()
    
    #dataset splitting, RandomSplitter function
    print("About to perform train/valid/test split.")
    splitter = RandomSplitter(verbosity=verbosity)
    print("Performing new split.")
    train_dataset,valid_dataset,test_dataset = splitter.train_valid_test_split(
                                datasets_nci, train_dir, valid_dir, test_dir)
    #running model
    nci_train,nci_valid = benchmarkTrainAndValid(base_dir,train_dataset,
                                                 valid_dataset,
                                                 tasks_nci,transformers_nci,
                                                 n_features,model,verbosity)
    time3 = time.time()
    
    with open('/home/zqwu/deepchem/examples/timing.csv','a') as f:
        f.write('Time for running nci,'+model+','+str(time3-time2)+'\n')
    with open('/home/zqwu/deepchem/examples/results.csv','a') as f:
      f.write ('\n'+'nci,train')
      for i in nci_train:
        f.write(','+i+','+str(nci_train[i]))
      f.write('\n'+'nci,valid')
      for i in nci_valid:
        f.write(','+i+','+str(nci_valid[i]))
      f.write('\n'+'nci,time_for_loading,'+str(time2-time1)+'seconds')
      f.write('\n'+'nci,time_for_running,'+str(time3-time2)+'seconds')

    del tasks_nci, datasets_nci, transformers_nci
    del train_dataset,valid_dataset,time1,time2,time3
  
  return None

def benchmarkTrainAndValid(base_dir,train_dataset,valid_dataset,tasks,
                           transformers,n_features = 1024,model = 'all',
                           verbosity = 'high'):
  train_scores = {}
  valid_scores = {}
  
  # Initialize metrics
  classification_metric = Metric(metrics.roc_auc_score, np.mean,
                                 verbosity=verbosity,
                                 mode="classification")
  
  assert model in ['all', 'tf', 'rf', 'keras']

  if model == 'all' or model == 'tf':
    # Initialize model folder
    model_dir_tf = os.path.join(base_dir, "model_tf")
    
    # Building tensorflow MultiTaskDNN model
    tensorflow_model = TensorflowMultiTaskClassifier(
        len(tasks), n_features, model_dir_tf, dropouts=[.25],
        learning_rate=0.001, weight_init_stddevs=[.1],
        batch_size=64, verbosity=verbosity)
    model_tf = TensorflowModel(tensorflow_model, model_dir_tf)
 
    print('-------------------------------------')
    print('Start fitting by tensorflow')
    model_tf.fit(train_dataset)
    train_evaluator = Evaluator(model_tf, train_dataset, transformers,
                                verbosity=verbosity)
    train_scores['tensorflow'] = train_evaluator.compute_model_performance(
                                [classification_metric])['mean-roc_auc_score']
    valid_evaluator = Evaluator(model_tf, valid_dataset, transformers,
                                verbosity=verbosity)
    valid_scores['tensorflow'] = valid_evaluator.compute_model_performance(
                                [classification_metric])['mean-roc_auc_score']

  
  if model == 'all' or model == 'rf':
    # Initialize model folder
    model_dir_rf = os.path.join(base_dir, "model_rf")
    
    # Building scikit random forest model
    def model_builder(model_dir_rf):
      sklearn_model = RandomForestClassifier(
        class_weight="balanced", n_estimators=500,n_jobs=-1)
      return SklearnModel(sklearn_model, model_dir_rf)
    model_rf = SingletaskToMultitask(tasks, model_builder, model_dir_rf)
    
    print('-------------------------------------')
    print('Start fitting by random forest')
    model_rf.fit(train_dataset)
    train_evaluator = Evaluator(model_rf, train_dataset, transformers, 
                                verbosity=verbosity)
    train_scores['random_forest'] = train_evaluator.compute_model_performance(
                                [classification_metric])['mean-roc_auc_score']
    valid_evaluator = Evaluator(model_rf, valid_dataset, transformers, 
                                verbosity=verbosity)
    valid_scores['random_forest'] = valid_evaluator.compute_model_performance(
                                [classification_metric])['mean-roc_auc_score']
  
  #currently not going to use this model
  '''
  if model == 'all' or model == 'keras':
    # Initialize model folder
    model_dir_kr = os.path.join(base_dir, "model_kr")
    # Building keras MultiTaskDNN model
    keras_model = MultiTaskDNN(len(tasks), n_features, "classification",
                               dropout=.25, learning_rate=.001, decay=1e-4)
    model_kr = KerasModel(keras_model, model_dir_kr, verbosity=verbosity)
      
    print('-------------------------------------')
    print('Start fitting by keras')
    model_kr.fit(train_dataset)
    train_evaluator = Evaluator(model_kr, train_dataset, transformers, 
                                verbosity=verbosity)
    train_scores['keras'] = train_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']
    valid_evaluator = Evaluator(model_kr, valid_dataset, transformers, 
                                verbosity=verbosity)
    valid_scores['keras'] = valid_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']
  '''
  
  return train_scores, valid_scores

if __name__ == '__main__':
  # Global variables
  np.random.seed(123)
  reload = True
  verbosity = 'high'
  
  #Working folder initialization
  base_dir = "/scratch/users/zqwu/benchmark_test_"+time.strftime(
                                                "%Y_%m_%d", time.localtime())
  if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
  os.makedirs(base_dir)
  
  #Datasets and models used in the benchmark test, all=all the datasets(models)
  datasetName = sys.argv[1]
  model = sys.argv[2]
  
  benchmarkLoadingDatasets(base_dir, n_features = 1024,
                           datasetName = datasetName, model = model,
                           reload = reload, verbosity = verbosity)
