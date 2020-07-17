"""
Script that trains graphconv models on delaney dataset.
"""
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc

import csv
from sklearn.metrics import r2_score
from deepchem.trans import undo_transforms

from delaney_datasets import load_delaney

MODEL_DIR = 'model_saves'
BATCH_SIZE = 64
LR = 1e-3
ERROR_BARS = True

delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
    featurizer='GraphConv', split='scaffold')
train_dataset, valid_dataset, test_dataset = delaney_datasets
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

model = dc.models.GraphConvModel(
    len(delaney_tasks),
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    use_queue=False,
    mode='regression',
    model_dir=MODEL_DIR,
    error_bars=ERROR_BARS)

model.fit(train_dataset, nb_epoch=8)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

model.save()
model.load_from_dir('model_saves')

mu, sigma = model.bayesian_predict(
    valid_dataset, transformers, untransform=True, n_passes=24)
print(mu[:4])
print(sigma[:4])

target = undo_transforms(valid_dataset.y, transformers)

print(r2_score(target, mu))

mu = mu[:, 0].tolist()
sigma = sigma[:, 0].tolist()
target = target[:, 0].tolist()

print(mu[:4])
print(sigma[:4])
print(target[:4])

in_one_sigma = 0
in_two_sigma = 0
in_four_sigma = 0

for i in xrange(0, len(mu)):
  if target[i] < (mu[i] + sigma[i]) and target[i] > (mu[i] - sigma[i]):
    in_one_sigma += 1
  if target[i] < (mu[i] + 2 * sigma[i]) and target[i] > (mu[i] - 2 * sigma[i]):
    in_two_sigma += 1
  if target[i] < (mu[i] + 4 * sigma[i]) and target[i] > (mu[i] - 4 * sigma[i]):
    in_four_sigma += 1

print('percent in 1 sigma [%f]' % (in_one_sigma / float(len(mu))))
print('percent in 2 sigma [%f]' % (in_two_sigma / float(len(mu))))
print('percent in 4 sigma [%f]' % (in_four_sigma / float(len(mu))))

print(sorted(sigma))
