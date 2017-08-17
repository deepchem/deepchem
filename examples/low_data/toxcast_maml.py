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

# Toxcast has data on 6874 molecules and 617 tasks.  However, the data is very
# sparse: most tasks do not include data for most molecules.  It also is very
# unbalanced: there are many more negatives than positives.  For every task,
# select 20 molecules (two batches of 10), where each batch has equal numbers
# of positives and negatives.

task_molecules = []
for i in range(n_tasks):
  positives = [j for j in range(n_molecules) if w[j, i] > 0 and y[j, i] == 1]
  negatives = [j for j in range(n_molecules) if w[j, i] > 0 and y[j, i] == 0]
  task_molecules.append(
      np.concatenate(
          [positives[:5], negatives[:5], positives[5:10], negatives[5:10]]))

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
    weights = np.ones((self.batch_size, 1))
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


def compute_auc(steps):
  maml.restore()
  y_true = []
  y_pred = []
  for task in range(learner.n_training_tasks, n_tasks):
    learner.set_task_index(task)
    maml.train_on_current_task(optimization_steps=steps)
    with model._get_tf("Graph").as_default():
      feed_dict = learner.get_batch()
      y_true.append(feed_dict[model.labels[0].out_tensor][:, 0, :])
      y_pred.append(
          maml._session.run(model.outputs[0], feed_dict=feed_dict)[:, 0, :])
  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_pred)
  return dc.metrics.compute_roc_auc_scores(y_true, y_pred)


print('AUC before fine tuning:', compute_auc(0))
print('AUC after fine tuning:', compute_auc(1))
