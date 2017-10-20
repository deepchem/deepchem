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

max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])

reshard_size = 512
transformer = deepchem.trans.DAGTransformer(max_atoms=max_atoms)
train_dataset.reshard(reshard_size)
train_dataset = transformer.transform(train_dataset)
valid_dataset.reshard(reshard_size)
valid_dataset = transformer.transform(valid_dataset)
test_dataset.reshard(reshard_size)
test_dataset = transformer.transform(test_dataset)

batch_size = 128
nb_epoch = 1000
learning_rate = 0.0005
n_graph_feat = 23

tf.set_random_seed(seed)
model = deepchem.models.DAGTensorGraph(
               1, 
               max_atoms=55,
               n_atom_feat=75,
               n_graph_feat=n_graph_feat,
               mode='regression',
               batch_size=batch_size,
               leanring_rate=learning_rate,
               use_queue=False)

model.fit(train_dataset, nb_epoch=nb_epoch)
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)
test_scores = model.evaluate(test_dataset, metric, transformers)

""" Expected Results:
  train_scores: {'mean-rms_score': 0.029829638487211169}
    
  valid_scores: {'mean-rms_score': 0.75142478279661051}
    
  test_scores: {'mean-rms_score': 0.53192168238754678}

"""