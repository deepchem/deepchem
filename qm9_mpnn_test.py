from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem
import numpy as np
import tensorflow as tf
import tempfile

seed = 123
data_dir = deepchem.utils.get_data_dir()
dataset_file = os.path.join(data_dir, "gdb9.sdf")

qm9_tasks = ["gap"]
featurizer = deepchem.feat.WeaveFeaturizer(graph_distance=False, explicit_H=True)

loader = deepchem.data.SDFLoader(
        tasks=qm9_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)

dataset = loader.featurize(dataset_file)
np.random.seed(seed)
splitter = deepchem.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
]
for transformer in transformers:
  train_dataset = transformer.transform(train_dataset)
  valid_dataset = transformer.transform(valid_dataset)
  test_dataset = transformer.transform(test_dataset)

metric = [deepchem.metrics.Metric(deepchem.metrics.mean_absolute_error, np.mean)]
  
batch_size = 64
nb_epoch = 100
learning_rate = 0.0003
n_hidden = 70
T = 2
M = 5

tf.set_random_seed(seed)
model_dir = tempfile.mkdtemp()
model = deepchem.models.DTNNTensorGraph(
    len(qm9_tasks),
    n_hidden=n_hidden,
    T=T,
    M=M,
    batch_size=batch_size,
    learning_rate=learning_rate,
    use_queue=False,
    mode="regression",
    model_dir=model_dir)
model.fit(train_dataset, nb_epoch=nb_epoch)
for rate in [learning_rate/5, learning_rate/20, learning_rate/100]:
  model = deepchem.models.DTNNTensorGraph(
      len(qm9_tasks),
      n_hidden=n_hidden,
      T=T,
      M=M,
      batch_size=batch_size,
      learning_rate=rate,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)
  model.restore()
  model.fit(train_dataset, nb_epoch=10)

train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)
test_scores = model.evaluate(test_dataset, metric, transformers)

model.fit(train_dataset, nb_epoch=10)
'''

'''