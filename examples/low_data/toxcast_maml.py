from __future__ import print_function

import deepchem as dc
import numpy as np
import random

# Load the data.

tasks, datasets, transformers = dc.molnet.load_toxcast()
(train_dataset, valid_dataset, test_dataset) = datasets
x = train_dataset.X
y = train_dataset.y
w = train_dataset.w
n_features = x.shape[1]
n_molecules = y.shape[0]
n_tasks = y.shape[1]

# For each task, create a list of the molecules for which we have data.

task_molecules = []
for i in range(n_tasks):
  task_molecules.append(w[:,i].nonzero()[0])
tasks = [i for i,m in enumerate(task_molecules) if len(m) >= 100]
random.shuffle(tasks)

# Create the model to train.

model = dc.models.TensorGraphMultiTaskClassifier(1, n_features, layer_sizes=[1000], dropouts=[0.0])
model.build()

# Define a MetaLearner describing the learning problem.

class ToxcastLearner(dc.metalearning.MetaLearner):
  def __init__(self):
    self.n_training_tasks = int(len(tasks)*0.8)
    self.batch_size = 50
    self.set_task_index(0)

  @property
  def loss(self):
    return model.loss

  def set_task_index(self, index):
    self.task_index = index
    self.task = tasks[index]
    self.batch_start = 0

  def select_task(self):
    self.set_task_index((self.task_index+1) % self.n_training_tasks)

  def get_batch(self):
    mols = task_molecules[self.task][self.batch_start:self.batch_start+self.batch_size]
    labels = np.zeros((self.batch_size, 1, 2))
    labels[np.arange(self.batch_size), 0, y[mols, self.task].astype(np.int64)] = 1
    weights = w[mols, self.task].reshape((-1, 1))
    feed_dict = {}
    feed_dict[model.features[0].out_tensor] = x[mols, :]
    feed_dict[model.labels[0].out_tensor] = labels
    feed_dict[model.task_weights[0].out_tensor] = weights
    self.batch_start += self.batch_size
    return feed_dict

# Run meta-learning on 80% of the tasks.

n_epochs = 40
learner = ToxcastLearner()
maml = dc.metalearning.MAML(learner)
steps = n_epochs*learner.n_training_tasks//maml.meta_batch_size
maml.fit(steps)

# Validate on the remaining tasks.

def compute_loss(steps):
  maml.restore()
  losses = []
  for task in range(learner.n_training_tasks, len(tasks)):
    learner.set_task_index(task)
    if steps > 0:
      maml.train_on_current_task(optimization_steps=steps)
    with model._get_tf("Graph").as_default():
      losses.append(maml._session.run(learner.loss, feed_dict=learner.get_batch()))
  return np.average(losses)

print('Loss before training:', compute_loss(0))
print('Loss after training:', compute_loss(1))
