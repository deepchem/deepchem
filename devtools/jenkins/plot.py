#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:49:01 2018

@author: zqwu
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

FILE = '/home/zqwu/deepchem/examples/results.csv'
path = FILE
dataset = 'delaney'
split = 'random'
      
ORDER = ['logreg', 'rf', 'rf_regression', 'xgb', 'xgb_regression', 
         'kernelsvm', 'krr', 'krr_ft', 'tf', 'tf_regression', 
         'tf_regression_ft', 'tf_robust', 'irv', 'graphconv',
         'graphconvreg', 'dag', 'dag_regression', 'ani', 'weave',
         'weave_regression', 'textcnn', 'textcnn_regression', 'dtnn', 'mpnn']

COLOR = {
  'logreg'
    }

def plot(dataset, split, path):
  data = {}
  with open(path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
      if line[0] == dataset and line[1] == split:
        data[line[3]] = line[8]
  labels = []
  values = []
  for model in ORDER:
    if model in data.keys():
      labels.append(model)
      values.append(float(data[model]))
  y_pos = np.arange(len(labels))
  plt.rcdefaults()
  fig, ax = plt.subplots()
  
  ax.barh(y_pos, values, align='center', color='green')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(labels)
  ax.invert_yaxis()
  ax.set_xlabel('R square')
  ax.set_xlim(left=0., right=1.)
  plt.show()
  
          