"""
Script that trains multitask models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

import deepchem as dc


def reshape_y(y):
  vectors = []
  for i in range(y.shape[1]):
    vectors.append(y[:, i])
  return vectors


# Only for debug!
np.random.seed(123)

metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets

inputs = tf.keras.layers.Input(shape=train_dataset.X[0].shape)
x = tf.keras.layers.Dense(1000, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = []
for i in range(len(tox21_tasks)):
  predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
  outputs.append(predictions)

optimizer = tf.keras.optimizers.Adam(lr=0.001)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(
    train_dataset.X,
    reshape_y(train_dataset.y),
    batch_size=50,
    epochs=1,
    verbose=False,
    sample_weight=reshape_y(train_dataset.w))

predictions = np.concatenate(model.predict(train_dataset.X), axis=1)
print(metric.compute_metric(train_dataset.y, predictions))

predictions = np.concatenate(model.predict(valid_dataset.X), axis=1)
print(metric.compute_metric(valid_dataset.y, predictions))
