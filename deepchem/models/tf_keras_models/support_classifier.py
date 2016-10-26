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

def get_task_dataset_minus_support(dataset, support, task):
  """Gets data for specified task, minus support points.

  Useful for evaluating model performance once trained (so that
  test compounds can be ensured distinct from support.)

  Parameters
  ----------
  dataset: deepchem.datasets.Dataset
    Source dataset.
  support: deepchem.datasets.Dataset
    The support dataset
  task: int
    Task number of task to select.
  """
  support_ids = set(support.ids)
  non_support_inds = [ind for ind in range(len(dataset))
                      if dataset.ids[ind] not in support_ids]

  # Remove support indices
  X = dataset.X[non_support_inds]
  y = dataset.y[non_support_inds]
  w = dataset.w[non_support_inds]
  ids = dataset.ids[non_support_inds]

  # Get task specific entries
  w_task = w[:, task]
  X_task = X[w_task != 0]
  y_task = y[w_task != 0, task]
  ids_task = ids[w_task != 0]
  # Now just get weights for this task
  w_task = w[w_task != 0, task]

  return NumpyDataset(X_task, y_task, w_task, ids_task)

def get_task_dataset(dataset, task):
  """Selects out entries for a particular task."""
  X, y, w, ids = dataset.X, dataset.y, dataset.w, dataset.ids
  # Get task specific entries
  w_task = w[:, task]
  X_task = X[w_task != 0]
  y_task = y[w_task != 0, task]
  ids_task = ids[w_task != 0]
  # Now just get weights for this task
  w_task = w[w_task != 0, task]

  return NumpyDataset(X_task, y_task, w_task, ids_task)

def get_task_test(dataset, batch_size, task, replace=True):
  """Gets test set from specified task.

  Samples random subset of size batch_size from specified task of dataset.
  Ensures that sampled points have measurements for this task.
  """
  w_task = dataset.w[:, task]
  X_task = dataset.X[w_task != 0]
  y_task = dataset.y[w_task != 0]
  ids_task = dataset.ids[w_task != 0]
  # Now just get weights for this task
  w_task = dataset.w[w_task != 0]

  inds = np.random.choice(np.arange(len(X_task)), batch_size, replace=replace)
  X_batch = X_task[inds]
  y_batch = np.squeeze(y_task[inds, task])
  w_batch = np.squeeze(w_task[inds, task])
  ids_batch = ids_task[inds]
  return NumpyDataset(X_batch, y_batch, w_batch, ids_batch)

def get_task_support(dataset, n_pos, n_neg, task, replace=True):
  """Generates a support set purely for specified task.
  
  Parameters
  ----------
  datasets: deepchem.datasets.Dataset
    Dataset from which supports are sampled.
  n_pos: int
    Number of positive samples in support.
  n_neg: int
    Number of negative samples in support.
  task: int
    Index of current task.
  replace: bool, optional
    Whether or not to use replacement when sampling supports.

  Returns
  -------
  list
    List of NumpyDatasets, each of which is a support set.
  """
  y_task = dataset.y[:, task]

  # Split data into pos and neg lists.
  pos_mols = np.where(y_task == 1)[0]
  neg_mols = np.where(y_task == 0)[0]

  # Get randomly sampled pos/neg indices (with replacement)
  pos_inds = pos_mols[np.random.choice(len(pos_mols), (n_pos), replace=replace)]
  neg_inds = neg_mols[np.random.choice(len(neg_mols), (n_neg), replace=replace)]

  # Handle one-d vs. non one-d feature matrices
  one_dimensional_features = (len(dataset.X.shape) == 1)
  if not one_dimensional_features:
    X_trial = np.vstack(
        [dataset.X[pos_inds], dataset.X[neg_inds]])
  else:
    X_trial = np.concatenate(
        [dataset.X[pos_inds], dataset.X[neg_inds]])
  y_trial = np.concatenate(
      [dataset.y[pos_inds, task], dataset.y[neg_inds, task]])
  w_trial = np.concatenate(
      [dataset.w[pos_inds, task], dataset.w[neg_inds, task]])
  ids_trial = np.concatenate(
      [dataset.ids[pos_inds], dataset.ids[neg_inds]])
  return NumpyDataset(X_trial, y_trial, w_trial, ids_trial)

class SupportGenerator(object):
  """ Generate support sets from a dataset.

  Iterates over tasks and trials. For each trial, picks one support from
  each task, and returns in a randomized order
  """
  def __init__(self, dataset, tasks, n_pos, n_neg, n_trials, replace):
    self.tasks = tasks
    self.n_tasks = len(tasks)
    self.n_trials = n_trials
    self.dataset = dataset
    self.n_pos = n_pos
    self.n_neg = n_neg
    self.replace = replace

    # Init the iterator
    self.perm_tasks = np.random.permutation(self.tasks)
    # Set initial iterator state
    self.task_num = 0
    self.trial_num = 0

  def __iter__(self):
    return self

  # TODO(rbharath): This is generating data from one task at a time. Is it
  # wrong to have batches that mix information from multiple tasks?
  def next(self):
    """Sample next support.

    Supports are sampled from the tasks in a random order. Each support is
    drawn entirely from within one task.
    """
    if self.trial_num == self.n_trials:
      raise StopIteration
    else:
      task = self.perm_tasks[self.task_num]  # Get id from permutation
      #support = self.supports[task][self.trial_num]
      support = get_task_support(
          self.dataset, n_pos=self.n_pos, n_neg=self.n_neg, task=task,
          replace=self.replace)
      # Increment and update logic
      self.task_num += 1
      if self.task_num == self.n_tasks:
        self.task_num = 0  # Reset
        self.perm_tasks = np.random.permutation(self.tasks)  # Permute again
        self.trial_num += 1  # Upgrade trial index

      return (task, support)

  __next__ = next # Python 3.X compatibility

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
    self.pred_op, self.scores_op, self.loss_op = self.add_training_loss()
    # Get train function
    self.add_optimizer()

    # Initialize
    self.init_fn = tf.initialize_all_variables()
    sess.run(self.init_fn)  

  def add_optimizer(self):
    if self.optimizer_type == "adam":
      self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    else:
      raise ValueError("Optimizer type not recognized.")

    # Get train function
    self.train_op = self.optimizer.minimize(self.loss_op)

  def build(self):
    # Create target inputs
    self.test_label_placeholder = Input(
        #tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        name="label_placeholder"))
    self.test_weight_placeholder = Input(
        #tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        tensor=K.placeholder(shape=(self.test_batch_size), dtype='float32',
        name="weight_placeholder"))

    # TODO(rbharath): There should be weights for the support being used! 
    # Support labels
    self.support_label_placeholder = Input(
        tensor=K.placeholder(shape=[self.support_batch_size], dtype='float32',
        name="support_label_placeholder"))

  def construct_feed_dict(self, test, support, training=True, add_phase=False):
    """Constructs tensorflow feed from test/support sets."""
    # Generate dictionary elements for support 
    feed_dict = (
        self.model.support_graph_topology.batch_to_feed_dict(support.X))
    feed_dict[self.support_label_placeholder] = np.squeeze(support.y)
    # Get graph information for test 
    batch_topo_dict = (
        self.model.test_graph_topology.batch_to_feed_dict(test.X))
    feed_dict = merge_dicts([batch_topo_dict, feed_dict])
    # Generate dictionary elements for test
    feed_dict[self.test_label_placeholder] = np.squeeze(test.y)
    feed_dict[self.test_weight_placeholder] = np.squeeze(test.w)

    # Get information for keras 
    if add_phase:
      feed_dict[K.learning_phase()] = training
    return feed_dict

  def fit(self, dataset, n_trials_per_epoch=1000, nb_epoch=10, n_pos=1,
          n_neg=9, replace=True, **kwargs):
    """Fits model on dataset.

    Note that fitting for support models is quite different from fitting
    for other deep models. Fitting is a two-level process. During each epoch,
    we repeat n_trials_per_epoch, where for each trial, we randomply sample
    a support set for a given task, and independently a test set from that same
    task. The SupportGenerator class iterates over the tasks in random order.

    # TODO(rbharath): Is the concept of an epoch even meaningful here? There's
    # never a guarantee that the full dataset is covered as in usual fit.

    # TODO(rbharath): Would it improve performance to sample multiple test sets
    for each support set or would that only harm performance?
    
    # TODO(rbharath): Should replace be an option for sampling the test sets?

    Parameters
    ----------
    dataset: deepchem.datasets.Dataset
      Dataset to fit model on.
    n_trials_per_epoch: int, optional
      Number of (support, test) pairs to sample and train on per epoch.
    nb_epoch: int, optional
      Number of training epochs.
    n_pos: int, optional
      Number of positive examples per support.
    n_neg: int, optional
      Number of negative examples per support.
    replace: bool, optional
      Whether or not to use replacement when sampling supports/tests.
    """
    # Perform the optimization
    for epoch in range(nb_epoch):
      # TODO(rbharath): Try removing this learning rate.
      lr = self.learning_rate / (1 + float(epoch) / self.decay_T)
      print("Training epoch %d" % epoch)

      # Create different support sets
      for (task, support) in SupportGenerator(dataset, range(self.n_tasks),
          n_pos, n_neg, n_trials_per_epoch, replace):
        print("Sampled Support set")
        # Get batch to try it out on
        test = get_task_test(dataset, self.test_batch_size, task, replace)
        print("Obtained batch")
        feed_dict = self.construct_feed_dict(test, support)
        # Train on support set, batch pair
        self.sess.run(self.train_op, feed_dict=feed_dict)

  def save(self):
    """Save all models

    TODO(rbharath): Saving is not yet supported for this model.
    """
    pass

  def add_training_loss(self):
    """Adds training loss and scores for network."""
    pred, scores = self.get_scores()
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        scores, self.test_label_placeholder)
    weighted_losses = tf.mul(losses, self.test_weight_placeholder)
    loss = tf.reduce_sum(weighted_losses)

    return pred, scores, loss

  def get_scores(self):
    """Adds tensor operations for computing scores.

    Computes prediction yhat (eqn (1) in Matching networks) of class for test
    compounds.
    """
    # Get featurization for test 
    # Shape (n_test, n_feat)
    test_feat = self.model.get_test_output()  
    # Get featurization for support
    # Shape (n_support, n_feat)
    support_feat = self.model.get_support_output()  

    # Computes the inner part c() of the kernel
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
      max_dist_sq = 20
      g = -tf.maximum(tf.reduce_sum(tf.square(test_feat - support_feat), 2), max_dist_sq)
    # Note that gram matrix g has shape (n_test, n_support)

    # soft corresponds to a(xhat, x_i) in eqn (1) of Matching Networks paper 
    # https://arxiv.org/pdf/1606.04080v1.pdf
    # Computes softmax across axis 1, (so sums distances to support set for
    # each test entry)
    # Shape (n_test, n_support)
    soft = tf.nn.softmax(g)  # Renormalize

    # Weighted sum of support labels
    # Shape (n_support, 1)
    support_labels = tf.expand_dims(self.support_label_placeholder, 1)
    # pred is yhat in eqn (1) of Matching Networks.
    # Shape squeeze((n_test, n_support) * (n_support, 1)) = (n_test,)
    pred = tf.squeeze(tf.matmul(soft, support_labels), [1])

    # Clip softmax probabilities to range [epsilon, 1-epsilon]
    # Shape (n_test,)
    pred = tf.clip_by_value(pred, K.epsilon(), 1.-K.epsilon())

    # Convert to logit space using inverse sigmoid (logit) function
    # logit function: log(pred) - log(1-pred)
    # Used to invoke tf.nn.sigmoid_cross_entropy_with_logits
    # in Cross Entropy calculation.
    # Shape (n_test,)
    scores = tf.log(pred) - tf.log(tf.constant(1., dtype=tf.float32)-pred)

    return pred, scores

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

  def predict_proba(self, support, test):
    """Makes predictions on test given support.

    TODO(rbharath): Does not currently support any transforms.
    TODO(rbharath): Only for 1 task at a time currently. Is there a better way?
    """
    y_preds = []
    for (X_batch, y_batch, w_batch, ids_batch) in test.iterbatches(
        self.test_batch_size, deterministic=True):
      test_batch = NumpyDataset(X_batch, y_batch, w_batch, ids_batch)
      y_pred_batch = self.predict_proba_on_batch(support, test_batch)
      y_preds.append(y_pred_batch)
    y_pred = np.concatenate(y_preds)
    return y_pred

  def predict_on_batch(self, support, test_batch):
    """Make predictions on batch of data."""
    n_samples = len(test_batch)
    padded_test_batch = NumpyDataset(*pad_batch(
        self.test_batch_size, test_batch.X, test_batch.y, test_batch.w,
        test_batch.ids))
    feed_dict = self.construct_feed_dict(padded_test_batch, support)
    # Get scores
    pred, scores = self.sess.run([self.pred_op, self.scores_op], feed_dict=feed_dict)
    y_pred_batch = np.round(scores)
    return y_pred_batch

  def predict_proba_on_batch(self, support, test_batch):
    """Make predictions on batch of data."""
    n_samples = len(test_batch)
    padded_test_batch = NumpyDataset(*pad_batch(
        self.test_batch_size, test_batch.X, test_batch.y, test_batch.w,
        test_batch.ids))
    feed_dict = self.construct_feed_dict(padded_test_batch, support)
    # Get scores
    pred, scores = self.sess.run([self.pred_op, self.scores_op], feed_dict=feed_dict)
    y_pred_batch = to_one_hot(np.round(pred))
    return y_pred_batch
    
  def evaluate(self, dataset, test_tasks, metric, n_pos=1,
               n_neg=9, n_trials=1000, exclude_support=True, replace=True):
    """Evaluate performance of dataset on test_tasks according to metrics


    Evaluates the performance of the trained model by sampling supports randomly
    for each task in test_tasks. For each sampled support, the accuracy of the
    model with support provided is computed on all data for that task. If
    exclude_support is True (by default), the support set is excluded from this
    accuracy calculation. exclude_support should be set to false if model's
    memorization capacity wants to be evaluated. 
    

    Since the accuracy on a task is dependent on the choice of random support,
    the evaluation experiment is repeated n_trials times over all test_tasks.
    (Each task gets n_trials/len(test_tasks) experiments). The computed accuracies
    are averaged across trials.

    TODO(rbharath): Currently does not support any transformers.

    Parameters
    ----------
    dataset: deepchem.datasets.Dataset
      Dataset to test on.
    test_tasks: list
      List of task indices (list[int])
    metrics: deepchem.metrics.Metric
      Evaluation metric.
    n_pos: int, optional
      Number of positive samples per support.
    n_neg: int, optional
      Number of negative samples per support.
    exclude_support: bool, optional
      Whether support set should be excluded when computing model accuracy.
    replace: bool, optional
      Whether or not to use replacement when sampling supports.
    """
    # Get batches
    task_scores = {task: [] for task in test_tasks}
    for (task, support) in SupportGenerator(dataset, test_tasks,
         n_pos, n_neg, n_trials, replace):
      print("Sampled Support set.")
      if exclude_support:
        print("Removing support datapoints for eval.")
        task_dataset = get_task_dataset_minus_support(dataset, support, task)
      else:
        print("Keeping support datapoints for eval.")
        task_dataset = get_task_dataset(dataset, task)
      y_pred = self.predict_proba(support, task_dataset)

      task_scores[task].append(metric.compute_metric(
          task_dataset.y, y_pred, task_dataset.w))

    # Join information for all tasks.
    mean_task_scores = {}
    for task in test_tasks:
      mean_task_scores[task] = np.mean(np.array(task_scores[task]))
    return mean_task_scores
