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

from muv.muv_datasets import load_muv
from nci.nci_datasets import load_nci
from pcba.pcba_datasets import load_pcba
from tox21.tox21_datasets import load_tox21

def benchmarkLoadingDatasets(base_dir, n_features = 1024, datasetName = 'all',
                             model = 'all', reload = True, verbosity = 'high'):
  Results = {}
  assert datasetName in ['all', 'muv', 'nci', 'pcba', 'tox21']
  
  if datasetName in ['all','muv']:
    print('-------------------------------------')
    print('Benchmark test on datasets: muv')
    print('-------------------------------------')
    tasks_muv, datasets_muv, transformers_muv = load_muv(base_dir, reload=reload)
    Results['muv_train'], Results['muv_valid'] = benchmarkTrainAndValid(base_dir, 
                                            datasets_muv, tasks_muv, transformers_muv, 
                                            n_features, model, verbosity)
    del tasks_muv, datasets_muv, transformers_muv
    
  if datasetName in ['all','nci']:
    print('-------------------------------------')
    print('Benchmark test on datasets: nci')
    print('-------------------------------------')
    tasks_nci, datasets_nci, transformers_nci = load_nci(base_dir, reload=reload)
    Results['nci_train'], Results['nci_valid'] = benchmarkTrainAndValid(base_dir, 
                                            datasets_nci, tasks_nci, transformers_nci, 
                                            n_features, model, verbosity)
    del tasks_nci, datasets_nci, transformers_nci
    
  if datasetName in ['all','pcba']:
    print('-------------------------------------')
    print('Benchmark test on datasets: pcba')
    print('-------------------------------------')
    tasks_pcba, datasets_pcba, transformers_pcba = load_pcba(base_dir, reload=reload)
    Results['pcba_train'], Results['pcba_valid'] = benchmarkTrainAndValid(base_dir, 
                                            datasets_pcba, tasks_pcba, transformers_pcba, 
                                            n_features, model, verbosity)
    del tasks_pcba, datasets_pcba, transformers_pcba
    
  if datasetName in ['all','tox21']:
    print('-------------------------------------')
    print('Benchmark test on datasets: tox21')
    print('-------------------------------------')
    tasks_tox21, datasets_tox21, transformers_tox21 = load_tox21(base_dir, reload=reload)
    Results['tox21_train'], Results['tox21_valid'] = benchmarkTrainAndValid(base_dir, 
                                            datasets_tox21, tasks_tox21, transformers_tox21, 
                                            n_features, model, verbosity)
    del tasks_tox21, datasets_tox21, transformers_tox21
    
  return Results

def benchmarkTrainAndValid(base_dir, datasets, tasks, transformers, 
                           n_features = 1024, model = 'all', verbosity = 'high'):
  train_scores = {}
  valid_scores = {}

  # Spliting datasets
  train_dataset, valid_dataset = datasets
  
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
    train_evaluator = Evaluator(model_tf, train_dataset, transformers, verbosity=verbosity)
    train_scores['tensorflow'] = train_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']
    valid_evaluator = Evaluator(model_tf, valid_dataset, transformers, verbosity=verbosity)
    valid_scores['tensorflow'] = valid_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']

  if model == 'all' or model == 'rf':
    # Initialize model folder
    model_dir_rf = os.path.join(base_dir, "model_rf")
    
    # Building scikit random forest model
    def model_builder(model_dir_rf):
      sklearn_model = RandomForestClassifier(
        class_weight="balanced", n_estimators=500)
      return SklearnModel(sklearn_model, model_dir_rf)
    model_rf = SingletaskToMultitask(tasks, model_builder, model_dir_rf)
    
    print('-------------------------------------')
    print('Start fitting by random forest')
    model_rf.fit(train_dataset)
    train_evaluator = Evaluator(model_rf, train_dataset, transformers, verbosity=verbosity)
    train_scores['random_forest'] = train_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']
    valid_evaluator = Evaluator(model_rf, valid_dataset, transformers, verbosity=verbosity)
    valid_scores['random_forest'] = valid_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']
  
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
    train_evaluator = Evaluator(model_kr, train_dataset, transformers, verbosity=verbosity)
    train_scores['keras'] = train_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']
    valid_evaluator = Evaluator(model_kr, valid_dataset, transformers, verbosity=verbosity)
    valid_scores['keras'] = valid_evaluator.compute_model_performance([classification_metric])['mean-roc_auc_score']

  return train_scores, valid_scores

if __name__ == '__main__':
  # Global variables
  np.random.seed(123)
  reload = True
  verbosity = 'high'
  
  #Working folder initialization
  base_dir = "/tmp/benchmark_test_"+time.strftime("%Y_%m_%d", time.localtime())
  if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
  os.makedirs(base_dir)
  
  #Datasets and models used in the benchmark test, all = all the datasets(models)
  datasetName = 'tox21'
  model = 'keras'
  
  Results = benchmarkLoadingDatasets(base_dir, n_features = 1024, datasetName = datasetName,
                             model = model, reload = reload, verbosity = verbosity)