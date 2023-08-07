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
from model import DRModel, DRAccuracy, ConfusionMatrix, QuadWeightedKappa
from data import load_images_DR

train, valid, test = load_images_DR(split='random', seed=123)
# Define and build model
model = DRModel(
    n_init_kernel=32,
    batch_size=32,
    learning_rate=1e-5,
    augment=True,
    model_dir='./test_model')
if not os.path.exists('./test_model'):
  os.mkdir('test_model')
model.build()
#model.restore()
metrics = [
    dc.metrics.Metric(DRAccuracy, mode='classification'),
    dc.metrics.Metric(QuadWeightedKappa, mode='classification')
]
cm = [dc.metrics.Metric(ConfusionMatrix, mode='classification')]

logger = logging.getLogger('deepchem.models.tensorgraph.tensor_graph')
logger.setLevel(logging.DEBUG)
for i in range(10):
  model.fit(train, nb_epoch=10, checkpoint_interval=3512)
  model.evaluate(train, metrics)
  model.evaluate(valid, metrics)
  model.evaluate(valid, cm)
  model.evaluate(test, metrics)
  model.evaluate(test, cm)
