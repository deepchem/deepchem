import numpy as np
import tensorflow as tf
import sys, os, pickle
from keras.engine import Layer
from keras.layers import Input
from keras import initializations, activations
from keras import backend as K
import sklearn.metrics
from deepchem.models import Model
from deepchem.datasets import pad_features
from deepchem.datasets import pad_batch
from deepchem.datasets import NumpyDataset
from deepchem.utils.evaluate import Evaluator
from deepchem.metrics import to_one_hot
from deepchem.models.tf_keras_models.graph_topology import merge_dicts

def get_single_task_test(dataset, batch_size, task):
  w_task = dataset.w[:, task]
  X_task = dataset.X[w_task != 0]
  y_task = dataset.y[w_task != 0]
  ids_task = dataset.ids[w_task != 0]
  # Now just get weights for this task
  w_task = dataset.w[w_task != 0]

  inds = np.random.choice(np.arange(len(X_task)), batch_size)
  X_batch = X_task[inds]
  y_batch = np.squeeze(y_task[inds, task])
  w_batch = np.squeeze(w_task[inds, task])
  ids_batch = ids_task[inds]
  return NumpyDataset(X_batch, y_batch, w_batch, ids_batch)

def get_task_dataset_minus_support(dataset, support, task):
  """Gets data for specified task, minus support points."""
  w_task = dataset.w[:, task]
  X_task = dataset.X[w_task != 0]
  y_task = dataset.y[w_task != 0, task]
  ids_task = dataset.ids[w_task != 0]
  # Now just get weights for this task
  w_task = dataset.w[w_task != 0, task]

  # TODO(rbharath): Haven't implemented minus-support functionality!
  return NumpyDataset(X_task, y_task, w_task, ids_task)

def get_task_support(dataset, n_pos, n_neg, task, n_trials):
  """Generates a support set purely for specified task.
  """
  # Make a shallow copy of the molecules list to avoid rarranging the original list
  #!  needs to be a list of molecules from the task (refactorable)
  mol_list = dataset.ids 
  y_task = dataset.y[:, task]
  # Split data into pos and neg lists.

  # When reimlementing this, it will be much faster if these are stored
  # separately for fast access Depending on how you want to handle different
  # tasks, you may want to refactor task here, which allows you to compute
  # the support sets for multiple tasks simultaneously due to the fact that
  # the task data used to be stored in the mol objects, which is not the case
  # in the refactor
  pos_mol = np.where(y_task == 1)[0]
  neg_mol = np.where(y_task == 0)[0]

  n_pos_avail = len(pos_mol)
  n_neg_avail = len(neg_mol)

  # Ensure that there are examples to sample
  assert n_pos_avail >= n_pos
  assert n_neg_avail >= n_neg

  # Get randomly sampled indices (with replacement)
  pos_ids = np.random.choice(n_pos_avail, (n_trials, n_pos))
  neg_ids = np.random.choice(n_neg_avail, (n_trials, n_neg))

  supports = []
  for trial in range(n_trials):
    one_dimensional_features = (len(dataset.X.shape) == 1)
    if not one_dimensional_features:
      X_trial = np.vstack(
          [dataset.X[pos_ids[trial]], dataset.X[neg_ids[trial]]])
    else:
      X_trial = np.concatenate(
          [dataset.X[pos_ids[trial]], dataset.X[neg_ids[trial]]])
    y_trial = np.concatenate(
        [dataset.y[pos_ids[trial], task], dataset.y[neg_ids[trial], task]])
    w_trial = np.concatenate(
        [dataset.w[pos_ids[trial], task], dataset.w[neg_ids[trial], task]])
    ids_trial = np.concatenate(
        [dataset.ids[pos_ids[trial]], dataset.ids[neg_ids[trial]]])
    supports.append(NumpyDataset(X_trial, y_trial, w_trial, ids_trial))
  return supports

class SupportGenerator(object):
  """ Generate support sets from a dataset.

  Iterates over tasks and trials. For each trial, picks one support from
  each task, and returns in a randomized order

  TODO(rbharath): Need to make this generate supports on the fly in next()
  instead of precomputing supports.
  """
  def __init__(self, dataset, tasks, n_pos, n_neg, n_trials):
    self.tasks = tasks
    self.n_tasks = len(tasks)
    self.n_trials = n_trials

    # Generate batches
    self.build(dataset, n_pos, n_neg, n_trials)

  # TODO(rbharath): This really shouldn't be built up-front. Need to do this
  # on the fly to save time...
  def build(self, dataset, n_pos, n_neg, n_trials):
    # Generate batches
    self.supports= {}
    for task in self.tasks:
      print("Handling task %s" % str(task))
      self.supports[task] = get_task_support(
          dataset, n_pos=n_pos, n_neg=n_neg, task=task, n_trials=n_trials)

    # Init the iterator
    self.perm_tasks = np.random.permutation(self.tasks)
    self.task_num = 0
    self.trial_num = 0

  def __iter__(self):
    return self

  # TODO(rbharath): This is generating data from one task at a time. Why not
  # have batches that mix information from multiple tasks?
  def next(self):
    if self.trial_num == self.n_trials:
      raise StopIteration
    else:
      task = self.perm_tasks[self.task_num]  # Get id from permutation
      support = self.supports[task][self.trial_num]

      # Increment and update logic
      self.task_num += 1
      if self.task_num == self.n_tasks:
        self.task_num = 0  # Reset
        self.perm_tasks = np.random.permutation(self.tasks)  # Permute again
        self.trial_num += 1  # Upgrade trial index

      return (task, support)

class SupportGraphClassifier(Model):
  def __init__(self, sess, model, n_tasks, train_tasks, 
               test_batch_size=10, support_batch_size=10,
               final_loss='cross_entropy', learning_rate=.001, decay_T=20,
               optimizer_type="adam", similarity="euclidean",
               beta1=.9, beta2=.999, **kwargs):
    """Builds a support-based classifier.

    See https://arxiv.org/pdf/1606.04080v1.pdf for definition of support.

    Parameters
    ----------
    sess: tf.Session
      Session for this model
    model: SequentialSupportModel 
      Contains core layers in model. 
    n_pos: int
      Number of positive examples in support.
    n_neg: int
      Number of negative examples in support.
    n_tasks: int
      Number of different tasks to consider.
    train_tasks: list
      List of those tasks used for training.
    """
    self.sess = sess
    self.similarity = similarity
    self.optimizer_type = optimizer_type
    self.optimizer_beta1 = beta1 
    self.optimizer_beta2 = beta2 
    self.n_tasks = n_tasks
    self.final_loss = final_loss
    self.model = model  
    self.test_batch_size = test_batch_size
    self.support_batch_size = support_batch_size

    self.learning_rate = learning_rate
    self.decay_T = decay_T
    self.epsilon = K.epsilon()

    self.build()
    self.scores_op, self.loss_op = self.add_training_loss()
    # Get train function
    self.add_optimizer()


  def add_optimizer(self):
    if self.optimizer_type == "adam":
      self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    else:
      raise ValueError("Optimizer type not recognized.")

    # Get train function
    self.train_op = self.optimizer.minimize(self.loss_op)

  def build(self):
    # Create target inputs
    self.label_placeholder = Input(
        #tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        name="label_placeholder"))
    self.weight_placeholder = Input(
        #tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        name="weight_placeholder"))

    # TODO(rbharath): Should there be support weights here?
    # Support labels
    self.support_label_placeholder = Input(
        tensor=K.placeholder(shape=[self.support_batch_size], dtype='float32',
        name="support_label_placeholder"))

  def construct_feed_dict(self, test, support, training=True):
    """Constructs tensorflow feed from test/support sets."""
    # Generate dictionary elements for support 
    support_labels_dict = {self.support_label_placeholder: np.squeeze(support.y)}
    support_topo_dict = (
        self.model.graph_topology_support.batch_to_feed_dict(support.X))
    support_dict = merge_dicts([support_topo_dict, support_labels_dict])
  
    # Generate dictionary elements for test
    ########################################################### DEBUG
    print("test.y.shape, test.w.shape")
    print(test.y.shape, test.w.shape)
    ########################################################### DEBUG
    target_dict = {self.label_placeholder: np.squeeze(test.y),
                   self.weight_placeholder: np.squeeze(test.w)}
    # Get graph information for x
    batch_topo_dict = (
        self.model.graph_topology_test.batch_to_feed_dict(test.X))
    test_dict =  merge_dicts([batch_topo_dict, target_dict])

    test_support_dicts = merge_dicts([test_dict, support_dict])

    # Get information for keras 
    keras_dict = {K.learning_phase() : training}
    feed_dict = merge_dicts([test_support_dicts, keras_dict])
    return feed_dict

  def fit(self, dataset, n_trials_per_epoch=1000, nb_epoch=10, n_pos=1,
          n_neg=9, **kwargs):
    # Perform the optimization
    for epoch in range(nb_epoch):
      lr = self.learning_rate / (1 + float(epoch) / self.decay_T)

      # Create different support sets
      for (task, support) in SupportGenerator(dataset, dataset.get_task_names(),
          n_pos, n_neg, n_trials_per_epoch):
        print("Sampled Support set")
        # Get batch to try it out on
        test = get_single_task_test(dataset, self.test_batch_size, task)
        print("Obtained batch")
        feed_dict = self.construct_feed_dict(test, support)
        # Train on support set, batch pair
        self.sess.run(self.train_op, feed_dict=feed_dict)

  def add_training_loss(self):
    scores = self.get_scores()

    losses = tf.nn.sigmoid_cross_entropy_with_logits(scores, self.label_placeholder)
    weighted_losses = tf.mul(losses, self.weight_placeholder)
    loss = tf.reduce_sum(weighted_losses)

    return scores, loss

  def get_scores(self):
    """Adds tensor operations for computing scores.

    TODO(rbharath): Not clear what the mathematical function computed here is.
    What equations in the Matching Networks paper does this correspond to?
    """
    # Get featurization for x
    test_feat = self.model.get_test_output()  
    # Get featurization for support
    support_feat = self.model.get_support_output()  

    # TODO(rbharath): I believe this part computes the inner part c() of the kernel
    # (the inset equation in section 2.1.1 of Matching networks paper). 
    # Normalize
    if self.similarity == 'cosine':
      rnorm_test = tf.rsqrt(tf.reduce_sum(tf.square(test_feat), 1,
                         keep_dims=True)) + K.epsilon()
      rnorm_support = tf.rsqrt(tf.reduce_sum(tf.square(support_feat), 1,
                               keep_dims=True)) + K.epsilon()
      test_feat_normalized = test_feat * rnorm_test
      support_feat_normalized = support_feat * rnorm_support

      # Transpose for mul
      support_feat_normalized_t = tf.transpose(support_feat_normalized, perm=[1,0])  
      g = tf.matmul(test_feat_normalized, support_feat_normalized_t)  # Gram matrix
    elif self.similarity == 'euclidean':
      test_feat = tf.expand_dims(test_feat, 1)
      support_feat = tf.expand_dims(support_feat, 0)
      #g = -tf.sqrt(tf.maximum(
      #       tf.reduce_sum(tf.square(test_feat - support_feat), 2), 1e-6))
      #g = -tf.sqrt(tf.reduce_sum(tf.square(test_feat - support_feat), 2))
      max_dist_sq = 20
      g = -tf.maximum(tf.reduce_sum(tf.square(test_feat - support_feat), 2), max_dist_sq)
    # soft corresponds to a(xhat, x_i) in eqn (1) of Matching Networks paper 
    # https://arxiv.org/pdf/1606.04080v1.pdf
    soft = tf.nn.softmax(g)  # Renormalize

    # Weighted sum of support labels
    support_labels = tf.expand_dims(self.support_label_placeholder, 1)
    # Prediction yhat in eqn (1) of Matching Networks.
    pred = tf.squeeze(tf.matmul(soft, support_labels), [1])

    pred = tf.clip_by_value(pred, K.epsilon(), 1.-K.epsilon())

    # Convert to logit space using inverse sigmoid (logit) function
    # logit function: log(pred) - log(1-pred)
    # Not sure which is the best way to compute
    #scores = tf.sub(tf.log(pred),
    #                tf.log(tf.sub(tf.constant(1., dtype=tf.float32), pred)))
    #scores = -tf.log(tf.inv(pred)-tf.constant(1., dtype=tf.float32))
    scores = tf.log(pred) - tf.log(tf.constant(1., dtype=tf.float32)-pred)

    return scores

  def predict(self, support, test):
    """Makes predictions on test given support.

    TODO(rbharath): Does not currently support any transforms.
    TODO(rbharath): Only for 1 task at a time currently. Is there a better way?
    """
    y_preds = []
    for (X_batch, y_batch, w_batch, ids_batch) in test.iterbatches(
        self.test_batch_size, deterministic=True):
      test_batch = NumpyDataset(X_batch, y_batch, w_batch, ids_batch)
      y_pred_batch = self.predict_on_batch(support, test_batch)
      y_preds.append(y_pred_batch)
    y_pred = np.concatenate(y_preds)
    return y_pred

  def predict_on_batch(self, support, test_batch):
    """Make predictions on batch of data."""
    n_samples = len(test_batch)
    ################################################# DEBUG
    print("predict_on_batch()")
    print("test_batch.X.shape, test_batch.y.shape, test_batch.w.shape, test_batch.ids.shape")
    print(test_batch.X.shape, test_batch.y.shape, test_batch.w.shape, test_batch.ids.shape)
    ################################################# DEBUG
    padded_test_batch = NumpyDataset(*pad_batch(
        self.test_batch_size, test_batch.X, test_batch.y, test_batch.w,
        test_batch.ids))
    feed_dict = self.construct_feed_dict(padded_test_batch, support)
    # Get scores
    scores = self.sess.run(self.scores_op, feed_dict=feed_dict)
    ################################################# DEBUG
    print("predict_on_batch()")
    print("scores.shape")
    print(scores.shape)
    print("scores")
    print(scores)
    y_pred_batch = to_one_hot(np.round(scores))
    y_pred_batch = y_pred_batch[:n_samples]
    ################################################# DEBUG
    return y_pred_batch
    
  def evaluate(self, dataset, test_tasks, metrics, n_trials=1000):
    """Evaluate performance of dataset on test_tasks according to metrics

    TODO(rbharath): Currently does not support any transformers.

    Since the performance on a task is dependent on the choice of support,
    the evaluation experiment is repeated n_trials times. The output scores
    is averaged and returned.
    """
    # Get batches
    task_scores = {(task, metric.name): []
                   for task in test_tasks for metric in metrics}
    for (task, support) in SupportGenerator(dataset, test_tasks,
        self.n_pos, self.n_neg, n_trials):
      print("Sampled Support set.")
      task_dataset = get_task_dataset_minus_support(dataset, support, task)
      y_pred = self.predict(support, task_dataset)

    #TODO(rbharath): Fix this up so numbers are meaningfully averaged across trials.
      for metric in metrics:
        task_scores[(task, metric.name)].append(metric.compute_metric(
            task_dataset.y, y_pred, task_dataset.w))

    ## Join information for all tasks.
    #joint_task_scores = {}
    #for metric in metrics:
    #  task_array = np.zeros(len(n_tasks))
    #  for task in test_tasks:
    #    task_array[task] = np.mean(np.array(task_scores[(task, metric.name)]))
    #  joint_task_scores[metric.name] = task_array
    #return joint_task_scores
