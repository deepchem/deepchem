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

import sys
import os
import numpy as np
import shutil
import time
import deepchem as dc
import tensorflow as tf
import argparse
from keras import backend as K
import csv

from low_data.datasets import load_tox21_convmol
from low_data.datasets import load_muv_convmol
from low_data.datasets import load_sider_convmol


def low_data_benchmark_loading_datasets(hyper_parameters,
                                        cross_valid=False,
                                        dataset='tox21',
                                        model='siamese',
                                        split='task',
                                        out_path='.'):
  """
  Loading dataset for low data benchmark test
  
  Parameters
  ----------
  hyper_parameters : dict of list
      hyper parameters including batch size, learning rate, etc.

  cross_valid : boolean, optional (default=False)
      whether implement cross validation on datasets
  
  dataset : string, optional (default='tox21')
      choice of which dataset to use, should be: tox21, muv, sider
      
  model : string,  optional (default='siamese')
      choice of which model to use, should be: siamese, attn, res
  
  split : string,  optional (default='task')
      choice of splitter function, only task splitter is supported

  out_path : string, optional(default='.')
      path of result file
  """
  # Check input
  if dataset in ['muv', 'tox21', 'sider']:
    mode = 'classification'
  else:
    raise ValueError('Dataset not supported')

  if not model in ['siamese', 'attn', 'res']:
    raise ValueError('Model not supported')

  if not split in ['task']:
    raise ValueError('Only task splitter is supported')

  loading_functions = {
      'tox21': load_tox21_convmol,
      'muv': load_muv_convmol,
      'sider': load_sider_convmol
  }

  print('-------------------------------------')
  print('Low data benchmark %s on dataset: %s' % (model, dataset))
  print('-------------------------------------')
  time_start = time.time()
  #loading datasets
  tasks, all_dataset, transformers = loading_functions[dataset]()
  time_finish_loading = time.time()

  #defining splitter function
  splitters = {'task': dc.splits.TaskSplitter()}
  splitter = splitters[split]

  #running model
  for count_hp, hp in enumerate(hyper_parameters[model]):
    # Loading general settings
    # Number of folds for split 
    K = hp['K']
    n_feat = hp['n_feat']

    fold_datasets = splitter.k_fold_split(all_dataset, K)
    if cross_valid:
      num_iter = K  # K iterations for cross validation
    else:
      num_iter = 1
    for count_iter in range(num_iter):
      # Assembling train and valid datasets
      train_folds = fold_datasets[:num_iter - count_iter - 1] + fold_datasets[
          num_iter - count_iter:]
      train_dataset = dc.splits.merge_fold_datasets(train_folds)
      valid_dataset = fold_datasets[num_iter - count_iter - 1]

      time_start_fitting = time.time()
      valid_scores = low_data_benchmark_classification(
          train_dataset, valid_dataset, hp, n_feat, model=model)
      time_finish_fitting = time.time()
      with open(os.path.join(out_path, 'results.csv'), 'ab') as f:
        writer = csv.writer(f)
        output_line = [count_hp, count_iter, dataset, model, 'valid']
        for i in valid_scores:
          output_line.append(i)
          for count in valid_scores[i]:
            output_line.append(valid_scores[i][count])
        output_line.append('time_for_running')
        output_line.append(time_finish_fitting - time_start_fitting)
        writer.writerow(output_line)

  return None


def low_data_benchmark_classification(train_dataset,
                                      valid_dataset,
                                      hyper_parameters,
                                      n_features,
                                      model='siamese',
                                      seed=123):
  """
  Calculate low data benchmark performance
  
  Parameters
  ----------
  train_dataset : dataset struct
      loaded dataset, ConvMol struct, used for training
      
  valid_dataset : dataset struct
      loaded dataset, ConvMol struct, used for validation
  
  hyper_parameters : dict
      hyper parameters including batch size, learning rate, etc.
 
  n_features : integer
      number of features, or length of binary fingerprints
  
  model : string,  optional (default='siamese')
      choice of which model to use, should be: siamese, attn, res

  Returns
  -------
  scores : dict
	predicting results(AUC) on valid set

  """
  scores = {}

  # Initialize metrics
  classification_metric = dc.metrics.Metric(
      dc.metrics.roc_auc_score, np.mean, mode="classification")

  assert model in ['siamese', 'attn', 'res']

  # Loading hyperparameters
  # num positive/negative ligands
  n_pos = hyper_parameters['n_pos']
  n_neg = hyper_parameters['n_neg']
  # Set batch sizes for network
  test_batch_size = hyper_parameters['test_batch_size']
  support_batch_size = n_pos + n_neg
  # Model structure
  n_filters = hyper_parameters['n_filters']
  n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

  # Traning settings
  nb_epochs = hyper_parameters['nb_epochs']
  n_train_trials = hyper_parameters['n_train_trials']
  n_eval_trials = hyper_parameters['n_eval_trials']

  learning_rate = hyper_parameters['learning_rate']

  g = tf.Graph()
  sess = tf.Session(graph=g)
  K.set_session(sess)
  # Building graph convolution model
  with g.as_default():
    tf.set_random_seed(seed)
    support_graph = dc.nn.SequentialSupportGraph(n_features)

    for count, n_filter in enumerate(n_filters):
      support_graph.add(dc.nn.GraphConv(int(n_filter), activation='relu'))
      support_graph.add(dc.nn.GraphPool())

    for count, n_fcnode in enumerate(n_fully_connected_nodes):
      support_graph.add(dc.nn.Dense(int(n_fcnode), activation='tanh'))

    support_graph.add_test(
        dc.nn.GraphGather(test_batch_size, activation='tanh'))
    support_graph.add_support(
        dc.nn.GraphGather(support_batch_size, activation='tanh'))
    if model in ['siamese']:
      pass
    elif model in ['attn']:
      max_depth = hyper_parameters['max_depth']
      support_graph.join(
          dc.nn.AttnLSTMEmbedding(test_batch_size, support_batch_size,
                                  max_depth))
    elif model in ['res']:
      max_depth = hyper_parameters['max_depth']
      support_graph.join(
          dc.nn.ResiLSTMEmbedding(test_batch_size, support_batch_size,
                                  max_depth))

    with tf.Session() as sess:
      model_low_data = dc.models.SupportGraphClassifier(
          sess,
          support_graph,
          test_batch_size=test_batch_size,
          support_batch_size=support_batch_size,
          learning_rate=learning_rate)

      print('-------------------------------------')
      print('Start fitting by low data model: ' + model)
      # Fit trained model
      model_low_data.fit(
          train_dataset,
          nb_epochs=nb_epochs,
          n_episodes_per_epoch=n_train_trials,
          n_pos=n_pos,
          n_neg=n_neg,
          log_every_n_samples=50)
      # Evaluating graph convolution model
      scores[model] = model_low_data.evaluate(
          valid_dataset,
          classification_metric,
          n_pos,
          n_neg,
          n_trials=n_eval_trials)

  return scores


if __name__ == '__main__':
  # Global variables
  np.random.seed(123)

  parser = argparse.ArgumentParser(
      description='Deepchem benchmark: ' +
      'giving performances of different learning models on datasets')
  parser.add_argument(
      '-s',
      action='append',
      dest='splitter_args',
      default=[],
      help='Choice of splitting function: task')
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
  splitters = args.splitter_args
  models = args.model_args
  datasets = args.dataset_args
  cross_valid = args.cross_valid

  if len(splitters) == 0:
    splitters = ['task']
  if len(models) == 0:
    models = ['siamese', 'attn', 'res']
  if len(datasets) == 0:
    datasets = ['tox21', 'sider', 'muv']

  #input hyperparameters
  #tf: dropouts, learning rate, layer_sizes, weight initial stddev,penalty,
  #    batch_size
  hps = {}
  hps = {}
  hps['siamese'] = [{
      'K': 4,
      'n_feat': 75,
      'n_pos': 1,
      'n_neg': 1,
      'test_batch_size': 128,
      'n_filters': [64, 128, 64],
      'n_fully_connected_nodes': [128],
      'max_depth': 3,
      'nb_epochs': 1,
      'n_train_trials': 2000,
      'n_eval_trials': 20,
      'learning_rate': 1e-4
  }]
  hps['res'] = hps['siamese']
  hps['attn'] = hps['siamese']

  for split in splitters:
    for dataset in datasets:
      for model in models:
        low_data_benchmark_loading_datasets(
            hps,
            cross_valid=cross_valid,
            dataset=dataset,
            model=model,
            split=split,
            out_path='.')
