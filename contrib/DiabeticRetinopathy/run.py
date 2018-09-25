#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 06:12:11 2018

@author: zqwu
"""

import deepchem as dc
import numpy as np
import pandas as pd
import os
import logging
from model import DRModel, DRAccuracy, DRSensitivity, DRSpecificity, ConfusionMatrix

PATH = './data/train_cut/'

# Input images
image_names = os.listdir(PATH)
labels = dict(zip(*np.transpose(np.array(pd.read_csv('./data/trainLabels.csv')))))

np.random.seed(123)
np.random.shuffle(image_names)
# Use 80% of the data for training, the rest for validation
train_images = image_names[:int(0.8*len(image_names))]
test_images = image_names[int(0.8*len(image_names)):]

print("Number of training images: %d" % len(train_images))

train_labels = np.array([labels[os.path.splitext(n)[0][4:]] for n in train_images]).reshape((-1, 1))
train_image_paths = [os.path.join(PATH, n) for n in train_images]

test_labels = np.array([labels[os.path.splitext(n)[0][4:]] for n in test_images]).reshape((-1, 1))
test_image_paths = [os.path.join(PATH, n) for n in test_images]

# Generate class weights according to training set labels distribution
classes, cts = np.unique(train_labels, return_counts=True)
weight_ratio = dict(zip(classes, np.max(cts) / cts.astype(float)))
train_weights = np.array([weight_ratio[l[0]] for l in train_labels]).reshape((-1, 1))
test_weights = np.array([weight_ratio[l[0]] for l in test_labels]).reshape((-1, 1))
    
loader = dc.data.ImageLoader()
train_data = loader.featurize(train_image_paths, 
                              labels=train_labels, 
                              weights=train_weights, 
                              read_img=False)
test_data = loader.featurize(test_image_paths, 
                             labels=test_labels, 
                             weights=test_weights,
                             read_img=False)

# Define and build model
model = DRModel(n_init_kernel=32,
                batch_size=32, 
                learning_rate=1e-5,
                augment=True,
                model_dir='./test_model')
model.build()
#model.restore()
metrics = [dc.metrics.Metric(DRAccuracy, mode='classification'),
           dc.metrics.Metric(DRSensitivity, mode='classification'),
           dc.metrics.Metric(DRSpecificity, mode='classification')]
cm = [dc.metrics.Metric(ConfusionMatrix, mode='classification')]

logger = logging.getLogger('deepchem.models.tensorgraph.tensor_graph')
logger.setLevel(logging.DEBUG)
for i in range(10):
  model.fit(train_data, nb_epoch=10, checkpoint_interval=3512)
  # Evaluate every 10 epochs
  model.evaluate(train_data, metrics)
  model.evaluate(test_data, metrics)
  model.evaluate(test_data, cm)
