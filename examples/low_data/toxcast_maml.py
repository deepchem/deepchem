"""
Train Metalearning models on the ToxCast dataset.
"""

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

# Toxcast has data on 6874 molecules and 617 tasks.  However,
# the data is very sparse: most tasks do not include data for
# most molecules.  It also is very unbalanced: there are many
# more negatives than positives.  For each task, create a list
# of alternating positives and negatives so each batch will have
# equal numbers of both.

task_molecules = []
for i in range(n_tasks):
  positives = [j for j in range(n_molecules) if w[j, i] > 0 and y[j, i] == 1]
  negatives = [j for j in range(n_molecules) if w[j, i] > 0 and y[j, i] == 0]
  np.random.shuffle(positives)
  np.random.shuffle(negatives)
  mols = sum((list(m) for m in zip(positives, negatives)), [])
  task_molecules.append(mols)

# Define a MetaLearner describing the learning problem.
class ToxcastLearner(dc.metalearning.MetaLearner):

  def __init__(self):
    self.n_training_tasks = int(n_tasks * 0.8)
    self.batch_size = 10
    self.batch_start = [0] * n_tasks
    self.set_task_index(0)
    self.w1 = tf.Variable(
        np.random.normal(size=[n_features, 1000], scale=0.02), dtype=tf.float32)
    self.w2 = tf.Variable(
        np.random.normal(size=[1000, 1], scale=0.02), dtype=tf.float32)
    self.b1 = tf.Variable(np.ones(1000), dtype=tf.float32)
    self.b2 = tf.Variable(np.zeros(1), dtype=tf.float32)

  def compute_model(self, inputs, variables, training):
    x, y = [tf.cast(i, tf.float32) for i in inputs]
    w1, w2, b1, b2 = variables
    dense1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    logits = tf.matmul(dense1, w2) + b2
    output = tf.sigmoid(logits)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    return loss, [output]

  @property
  def variables(self):
    return [self.w1, self.w2, self.b1, self.b2]

  def set_task_index(self, index):
    self.task = index

  def select_task(self):
    self.set_task_index((self.task + 1) % self.n_training_tasks)

  def get_batch(self):
    task = self.task
    start = self.batch_start[task]
    mols = task_molecules[task][start:start + self.batch_size]
    labels = np.zeros((self.batch_size, 1))
    labels[np.arange(self.batch_size), 0] = y[mols, task]
    if start + 2 * self.batch_size > len(task_molecules[task]):
      self.batch_start[task] = 0
    else:
      self.batch_start[task] += self.batch_size
    return [x[mols, :], labels]


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
  for task in range(learner.n_training_tasks, n_tasks):
    learner.set_task_index(task)
    if optimize:
      maml.train_on_current_task(restore=True)
    inputs = learner.get_batch()
    loss, prediction = maml.predict_on_batch(inputs)
    y_true.append(inputs[1])
    y_pred.append(prediction[0][:, 0])
    losses.append(loss)
  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_pred)
  print()
  print('Cross entropy loss:', np.mean(losses))
  print('Prediction accuracy:', accuracy_score(y_true, y_pred > 0.5))
  print('ROC AUC:', dc.metrics.roc_auc_score(y_true, y_pred))
  print()


print('Before fine tuning:')
compute_scores(False)
print('After fine tuning:')
compute_scores(True)
