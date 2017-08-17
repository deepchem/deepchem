from __future__ import print_function

import deepchem as dc
import numpy as np

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
# Since we are interested in low data learning, we will only use the first
# 20 molecules with data for each task, even though most tasks have much
# more than that.

task_molecules = []
for i in range(n_tasks):
  task_molecules.append(w[:, i].nonzero()[0])

# Create the model to train.

model = dc.models.TensorGraphMultiTaskClassifier(
    1, n_features, layer_sizes=[1000], dropouts=[0.0])
model.build()

# Define a MetaLearner describing the learning problem.


class ToxcastLearner(dc.metalearning.MetaLearner):

  def __init__(self):
    self.n_training_tasks = int(n_tasks * 0.8)
    self.batch_size = 10
    self.set_task_index(0)

  @property
  def loss(self):
    return model.loss

  def set_task_index(self, index):
    self.task = index
    self.batch_start = 0

  def select_task(self):
    self.set_task_index((self.task + 1) % self.n_training_tasks)

  def get_batch(self):
    mols = task_molecules[self.task][self.batch_start:
                                     self.batch_start + self.batch_size]
    labels = np.zeros((self.batch_size, 1, 2))
    labels[np.arange(self.batch_size), 0, y[mols, self.task].astype(
        np.int64)] = 1
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
steps = n_epochs * learner.n_training_tasks // maml.meta_batch_size
maml.fit(steps)

# Validate on the remaining tasks.


def compute_loss(steps):
  maml.restore()
  y_true = []
  y_pred = []
  for task in range(learner.n_training_tasks, n_tasks):
    learner.set_task_index(task)
    if steps > 0:
      maml.train_on_current_task(optimization_steps=steps)
    with model._get_tf("Graph").as_default():
      feed_dict = learner.get_batch()
      y_true.append(feed_dict[model.labels[0].out_tensor][0])
      y_pred.append(maml._session.run(model.outputs[0], feed_dict=feed_dict)[0])
  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_pred)
  return dc.metrics.compute_roc_auc_scores(y_true, y_pred)


print('AUC before fine tuning:', compute_loss(0))
print('AUC after fine tuning:', compute_loss(1))
