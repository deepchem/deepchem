"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import json

np.random.seed(123)
import tensorflow as tf

tf.set_random_seed(123)
import deepchem as dc
from tox21_datasets import load_tox21
from deepchem.models.tensorgraph.models.graph_models import PetroskiSuchTensorGraph

model_dir = "/tmp/graph_conv"

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='AdjMatrix')
train_dataset, valid_dataset, test_dataset = tox21_datasets
print(train_dataset.data_dir)
print(valid_dataset.data_dir)

# Fit models
metric = dc.metrics.Metric(
  dc.metrics.roc_auc_score, np.mean, mode="classification")

# Batch size of models
batch_size = 128

model = PetroskiSuchTensorGraph(
  len(tox21_tasks), batch_size=batch_size, mode='classification')

best_train = 0.0
best_val = 0.0
loop_num = 0
model.fit(train_dataset, nb_epoch=10)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
ts = train_scores['mean-roc_auc_score']
if ts > best_train:
  best_train = ts

print("Validation scores")
print(valid_scores)
vs = valid_scores['mean-roc_auc_score']
if vs > best_val:
  best_val = vs
print("BEST SCORES")
print(best_train, best_val)
loop_num += 1
try:
  d = json.loads(open('spectral.json').read())
except:
  d = {"train_scores": [], "val_scores": []}
d['train_scores'].append(ts)
d['val_scores'].append(vs)
with open('spectral.json', 'w') as fout:
  fout.write(json.dumps(d))
