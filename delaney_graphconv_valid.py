from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import deepchem
import numpy as np
import tensorflow as tf

seed = 123
np.random.seed(seed)

tasks, datasets, transformers = deepchem.molnet.load_delaney(featurizer='GraphConv', split='random', reload=False)
train_dataset, valid_dataset, test_dataset = datasets
metric = [deepchem.metrics.Metric(deepchem.metrics.rms_score, np.mean)]
  
batch_size = 150
nb_epoch = 1000
learning_rate = 0.0008

tf.set_random_seed(seed)
model = deepchem.models.GraphConvTensorGraph(1, mode='regression',
                                             batch_size=batch_size,
                                             leanring_rate=learning_rate)

model.fit(train_dataset, nb_epoch=nb_epoch)
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)
test_scores = model.evaluate(test_dataset, metric, transformers)

""" Expected Results:
  train_scores: {'mean-rms_score': 0.058672648022210311}
    
  valid_scores: {'mean-rms_score': 0.3635136142334261}
    
  test_scores: {'mean-rms_score': 0.35664025829369983}

"""