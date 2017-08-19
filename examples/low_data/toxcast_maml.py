from __future__ import print_function

import deepchem as dc
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

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
# unbalanced: there are many more negatives than positives.  For each task,
# create a list of alternating postives and negatives so each batch will have
# equal numbers of both.

task_molecules = []
for i in range(n_tasks):
  positives = [j for j in range(n_molecules) if w[j, i] > 0 and y[j, i] == 1]
  negatives = [j for j in range(n_molecules) if w[j, i] > 0 and y[j, i] == 0]
  np.random.shuffle(positives)
  np.random.shuffle(negatives)
  mols = sum((list(x) for x in zip(positives, negatives)), [])
  task_molecules.append(mols)

# Create the model to train.  We use a simple fully connected network with
# one hidden layer.

model = dc.models.TensorGraphMultiTaskClassifier(
    1, n_features, layer_sizes=[1000], dropouts=[0.0])
model.build()

# Define a MetaLearner describing the learning problem.


class ToxcastLearner(dc.metalearning.MetaLearner):

  def __init__(self):
    self.n_training_tasks = int(n_tasks * 0.8)
    self.batch_size = 10
    self.batch_start = [0] * n_tasks
    self.set_task_index(0)

  @property
  def loss(self):
    return model.loss

  def set_task_index(self, index):
    self.task = index

  def select_task(self):
    self.set_task_index((self.task + 1) % self.n_training_tasks)

  def get_batch(self):
    task = self.task
    start = self.batch_start[task]
    mols = task_molecules[task][start:start + self.batch_size]
    labels = np.zeros((self.batch_size, 1, 2))
    labels[np.arange(self.batch_size), 0, y[mols, task].astype(np.int64)] = 1
    weights = np.ones((self.batch_size, 1))
    feed_dict = {}
    feed_dict[model.features[0].out_tensor] = x[mols, :]
    feed_dict[model.labels[0].out_tensor] = labels
    feed_dict[model.task_weights[0].out_tensor] = weights
    if start + 2 * self.batch_size > len(task_molecules[task]):
      self.batch_start[task] = 0
    else:
      self.batch_start[task] += self.batch_size
    return feed_dict


# Run meta-learning on 80% of the tasks.

n_epochs = 20
learner = ToxcastLearner()
maml = dc.metalearning.MAML(learner)
steps = n_epochs * learner.n_training_tasks // maml.meta_batch_size
maml.fit(steps)

# Validate on the remaining tasks.


def compute_scores(optimize):
  maml.restore()
  y_true = []
  y_pred = []
  losses = []
  with model._get_tf("Graph").as_default():
    prediction = tf.contrib.layers.softmax(model.outputs[0].out_tensor)
    for task in range(learner.n_training_tasks, n_tasks):
      learner.set_task_index(task)
      if optimize:
        maml.train_on_current_task()
      feed_dict = learner.get_batch()
      y_true.append(feed_dict[model.labels[0].out_tensor][:, 0, 0])
      y_pred.append(maml._session.run(prediction, feed_dict=feed_dict)[:, 0, 0])
      losses.append(maml._session.run(model.loss, feed_dict=feed_dict))
  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_pred)
  print()
  print('Cross entropy loss:', np.mean(losses))
  print('Prediction accuracy:', accuracy_score(y_true, y_pred > 0.5))
  print('ROC AUC:', dc.metrics.compute_roc_auc_scores(y_true, y_pred))
  print()


print('Before fine tuning:')
compute_scores(False)
print('After fine tuning:')
compute_scores(True)
