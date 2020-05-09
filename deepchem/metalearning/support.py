"""
Train support-based models.
"""
import warnings
import numpy as np
import tensorflow as tf
import sys
import time
from deepchem.models import Model
from deepchem.data import pad_batch
from deepchem.data import NumpyDataset
from deepchem.metrics import to_one_hot
from deepchem.metrics import from_one_hot
#from deepchem.models.tf_new_models.graph_topology import merge_dicts
#from deepchem.nn import model_ops
from deepchem.data import SupportGenerator
from deepchem.data import EpisodeGenerator
from deepchem.data import get_task_dataset
from deepchem.data import get_single_task_test
from deepchem.data import get_task_dataset_minus_support

def merge_dicts(l):
  """Convenience function to merge list of dictionaries."""
  merged = {}
  for dict in l:
    merged = merge_two_dicts(merged, dict)
  return merged

def cosine_distances(test, support):
  """Computes pairwise cosine distances between provided tensors

  Parameters
  ----------
  test: tf.Tensor
    Of shape (n_test, n_feat)
  support: tf.Tensor
    Of shape (n_support, n_feat)

  Returns
  -------
  tf.Tensor:
    Of shape (n_test, n_support)
  """
  rnorm_test = tf.rsqrt(
      tf.reduce_sum(tf.square(test), 1, keep_dims=True)) + 1e-7
  rnorm_support = tf.rsqrt(
      tf.reduce_sum(tf.square(support), 1, keep_dims=True)) + 1e-7
  test_normalized = test * rnorm_test
  support_normalized = support * rnorm_support

  # Transpose for mul
  support_normalized_t = tf.transpose(support_normalized, perm=[1, 0])
  g = tf.matmul(test_normalized, support_normalized_t)  # Gram matrix
  return g

class GraphTopology(object):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self, n_feat, name='topology', max_deg=10, min_deg=0):
    """
    Note that batch size is not specified in a GraphTopology object. A
    batch of molecules must be combined into a disconnected graph and
    fed to topology directly to handle batches.

    Parameters
    ----------
    n_feat: int
      Number of features per atom.
    name: str, optional
      Name of this manager.
    max_deg: int, optional
      Maximum #bonds for atoms in molecules.
    min_deg: int, optional
      Minimum #bonds for atoms in molecules.
    """
    warnings.warn("GraphTopology is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)

    #self.n_atoms = n_atoms
    self.n_feat = n_feat

    self.name = name
    self.max_deg = max_deg
    self.min_deg = min_deg

    self.atom_features_placeholder = tensor = tf.compat.v1.placeholder(
        dtype='float32',
        shape=(None, self.n_feat),
        name=self.name + '_atom_features')
    self.deg_adj_lists_placeholders = [
        tf.compat.v1.placeholder(
            dtype='int32',
            shape=(None, deg),
            name=self.name + '_deg_adj' + str(deg))
        for deg in range(1, self.max_deg + 1)
    ]
    self.deg_slice_placeholder = tf.compat.v1.placeholder(
        dtype='int32',
        shape=(self.max_deg - self.min_deg + 1, 2),
        name=self.name + '_deg_slice')
    self.membership_placeholder = tf.compat.v1.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_membership')

    # Define the list of tensors to be used as topology
    self.topology = [self.deg_slice_placeholder, self.membership_placeholder]
    self.topology += self.deg_adj_lists_placeholders

    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_input_placeholders(self):
    """All placeholders.

    Contains atom_features placeholder and topology placeholders.
    """
    return self.inputs

  def get_topology_placeholders(self):
    """Returns topology placeholders

    Consists of deg_slice_placeholder, membership_placeholder, and the
    deg_adj_list_placeholders.
    """
    return self.topology

  def get_atom_features_placeholder(self):
    return self.atom_features_placeholder

  def get_deg_adjacency_lists_placeholders(self):
    return self.deg_adj_lists_placeholders

  def get_deg_slice_placeholder(self):
    return self.deg_slice_placeholder

  def get_membership_placeholder(self):
    return self.membership_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of mol_graphs into tensorflow feed_dict.

    Assigns the graph information in array of ConvMol objects to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of ConvMol objects

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Merge mol conv objects
    batch = ConvMol.agglomerate_mols(batch)
    atoms = batch.get_atom_features()
    deg_adj_lists = [
        batch.deg_adj_lists[deg] for deg in range(1, self.max_deg + 1)
    ]

    # Generate dicts
    deg_adj_dict = dict(
        list(zip(self.deg_adj_lists_placeholders, deg_adj_lists)))
    atoms_dict = {
        self.atom_features_placeholder: atoms,
        self.deg_slice_placeholder: batch.deg_slice,
        self.membership_placeholder: batch.membership
    }
    return merge_dicts([atoms_dict, deg_adj_dict])

class SequentialGraph(object):
  """An analog of Keras Sequential class for Graph data.

  Like the Sequential class from Keras, but automatically passes
  topology placeholders from GraphTopology to each graph layer (from
  layers) added to the network. Non graph layers don't get the extra
  placeholders. 
  """

  def __init__(self, n_feat):
    """
    Parameters
    ----------
    n_feat: int
      Number of features per atom.
    """
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.graph_topology = GraphTopology(n_feat)
      self.output = self.graph_topology.get_atom_features_placeholder()
    # Keep track of the layers
    self.layers = []

  def add(self, layer):
    """Adds a new layer to model."""
    with self.graph.as_default():
      # For graphical layers, add connectivity placeholders
      if type(layer).__name__ in ['GraphConv', 'GraphGather', 'GraphPool']:
        if (len(self.layers) > 0 and hasattr(self.layers[-1], "__name__")):
          assert self.layers[-1].__name__ != "GraphGather", \
                  'Cannot use GraphConv or GraphGather layers after a GraphGather'

        self.output = layer([self.output] +
                            self.graph_topology.get_topology_placeholders())
      else:
        self.output = layer(self.output)

      # Add layer to the layer list
      self.layers.append(layer)

  def get_graph_topology(self):
    return self.graph_topology

  def get_num_output_features(self):
    """Gets the output shape of the featurization layers of the network"""
    return self.layers[-1].output_shape[1]

  def return_outputs(self):
    return self.output

  def return_inputs(self):
    return self.graph_topology.get_input_placeholders()

  def get_layer(self, layer_id):
    return self.layers[layer_id]

class SequentialSupportGraph(object):
  """An analog of Keras Sequential model for test/support models."""

  def __init__(self, n_feat):
    """
    Parameters
    ----------
    n_feat: int
      Number of atomic features.
    """
    warnings.warn("SequentialSupportWeaveGraph is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.graph = tf.Graph()
    with self.graph.as_default():
      # Create graph topology and x
      self.test_graph_topology = GraphTopology(n_feat, name='test')
      self.support_graph_topology = GraphTopology(n_feat, name='support')
      self.test = self.test_graph_topology.get_atom_features_placeholder()
      self.support = self.support_graph_topology.get_atom_features_placeholder()

    # Keep track of the layers
    self.layers = []
    # Whether or not we have used the GraphGather layer yet
    self.bool_pre_gather = True

  def add(self, layer):
    """Adds a layer to both test/support stacks.

    Note that the layer transformation is performed independently on the
    test/support tensors.
    """
    with self.graph.as_default():
      self.layers.append(layer)

      # Update new value of x
      if type(layer).__name__ in ['GraphConv', 'GraphGather', 'GraphPool']:
        assert self.bool_pre_gather, "Cannot apply graphical layers after gather."

        self.test = layer([self.test] + self.test_graph_topology.topology)
        self.support = layer([self.support] +
                             self.support_graph_topology.topology)
      else:
        self.test = layer(self.test)
        self.support = layer(self.support)

      if type(layer).__name__ == 'GraphGather':
        self.bool_pre_gather = False  # Set flag to stop adding topology

  def add_test(self, layer):
    """Adds a layer to test."""
    with self.graph.as_default():
      self.layers.append(layer)

      # Update new value of x
      if type(layer).__name__ in ['GraphConv', 'GraphPool', 'GraphGather']:
        self.test = layer([self.test] + self.test_graph_topology.topology)
      else:
        self.test = layer(self.test)

  def add_support(self, layer):
    """Adds a layer to support."""
    with self.graph.as_default():
      self.layers.append(layer)

      # Update new value of x
      if type(layer).__name__ in ['GraphConv', 'GraphPool', 'GraphGather']:
        self.support = layer([self.support] +
                             self.support_graph_topology.topology)
      else:
        self.support = layer(self.support)

  def join(self, layer):
    """Joins test and support to a two input two output layer"""
    with self.graph.as_default():
      self.layers.append(layer)
      self.test, self.support = layer([self.test, self.support])

  def get_test_output(self):
    return self.test

  def get_support_output(self):
    return self.support

  def return_outputs(self):
    return [self.test] + [self.support]

  def return_inputs(self):
    return (self.test_graph_topology.get_inputs() +
            self.support_graph_topology.get_inputs())

class SupportGraphClassifier(Model):

  def __init__(self,
               model,
               test_batch_size=10,
               support_batch_size=10,
               learning_rate=.001,
               similarity="cosine",
               **kwargs):
    """Builds a support-based classifier.

    See https://arxiv.org/pdf/1606.04080v1.pdf for definition of support.

    Parameters
    ----------
    sess: tf.Session
      Session for this model
    model: SequentialSupportGraph
      Contains core layers in model. 
    n_pos: int
      Number of positive examples in support.
    n_neg: int
      Number of negative examples in support.
    """
    warnings.warn("SupportGraphClassifier is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.similarity = similarity
    self.model = model
    self.sess = tf.Session(graph=self.model.graph)
    self.test_batch_size = test_batch_size
    self.support_batch_size = support_batch_size

    self.learning_rate = learning_rate
    self.epsilon = 1e-7

    with self.model.graph.as_default():
      self.add_placeholders()
      self.pred_op, self.scores_op, self.loss_op = self.add_training_loss()
      # Get train function
      self.train_op = self.get_training_op(self.loss_op)

      # Initialize
      self.init_fn = tf.global_variables_initializer()
      self.sess.run(self.init_fn)

  def get_training_op(self, loss):
    """Attaches an optimizer to the graph."""
    opt = tf.train.AdamOptimizer(self.learning_rate)
    return opt.minimize(self.loss_op, name="train")

  def add_placeholders(self):
    """Adds placeholders to graph."""
    #################################################################### DEBUG
    #self.test_label_placeholder = Input(
    #    tensor=tf.placeholder(dtype='float32', shape=(self.test_batch_size),
    #    name="label_placeholder"))
    #self.test_weight_placeholder = Input(
    #    tensor=tf.placeholder(dtype='float32', shape=(self.test_batch_size),
    #    name="weight_placeholder"))
    self.test_label_placeholder = tf.compat.v1.placeholder(
        dtype='float32', shape=(self.test_batch_size), name="label_placeholder")
    self.test_weight_placeholder = tf.compat.v1.placeholder(
        dtype='float32',
        shape=(self.test_batch_size),
        name="weight_placeholder")

    # TODO(rbharath): Should weights for the support be used?
    # Support labels
    #self.support_label_placeholder = Input(
    #    tensor=tf.placeholder(dtype='float32', shape=[self.support_batch_size],
    #    name="support_label_placeholder"))
    self.support_label_placeholder = tf.compat.v1.placeholder(
        dtype='float32',
        shape=[self.support_batch_size],
        name="support_label_placeholder")
    self.phase = tf.placeholder(dtype='bool', name='keras_learning_phase')
    #################################################################### DEBUG

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

    if add_phase:
      feed_dict[self.phase] = training
    return feed_dict

  def fit(self,
          dataset,
          n_episodes_per_epoch=1000,
          nb_epochs=1,
          n_pos=1,
          n_neg=9,
          log_every_n_samples=10,
          **kwargs):
    """Fits model on dataset using cached supports.

    For each epcoh, sample n_episodes_per_epoch (support, test) pairs and does
    gradient descent.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to fit model on.
    nb_epochs: int, optional
      number of epochs of training.
    n_episodes_per_epoch: int, optional
      Number of (support, test) pairs to sample and train on per epoch.
    n_pos: int, optional
      Number of positive examples per support.
    n_neg: int, optional
      Number of negative examples per support.
    log_every_n_samples: int, optional
      Displays info every this number of samples
    """
    time_start = time.time()
    # Perform the optimization
    n_tasks = len(dataset.get_task_names())
    n_test = self.test_batch_size

    feed_total, run_total = 0, 0
    for epoch in range(nb_epochs):
      # Create different support sets
      episode_generator = EpisodeGenerator(dataset, n_pos, n_neg, n_test,
                                           n_episodes_per_epoch)
      recent_losses = []
      for ind, (task, support, test) in enumerate(episode_generator):
        if ind % log_every_n_samples == 0:
          print("Epoch %d, Sample %d from task %s" % (epoch, ind, str(task)))
        # Get batch to try it out on
        feed_start = time.time()
        feed_dict = self.construct_feed_dict(test, support)
        feed_end = time.time()
        feed_total += (feed_end - feed_start)
        # Train on support set, batch pair
        run_start = time.time()
        _, loss = self.sess.run(
            [self.train_op, self.loss_op], feed_dict=feed_dict)
        run_end = time.time()
        run_total += (run_end - run_start)
        if ind % log_every_n_samples == 0:
          mean_loss = np.mean(np.array(recent_losses))
          print("\tmean loss is %s" % str(mean_loss))
          recent_losses = []
        else:
          recent_losses.append(loss)
    time_end = time.time()
    print("fit took %s seconds" % str(time_end - time_start))
    print("feed_total: %s" % str(feed_total))
    print("run_total: %s" % str(run_total))

  def save(self):
    """Save all models

    TODO(rbharath): Saving is not yet supported for this model.
    """
    pass

  def add_training_loss(self):
    """Adds training loss and scores for network."""
    pred, scores = self.get_scores()
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=scores, labels=self.test_label_placeholder)
    weighted_losses = tf.multiply(losses, self.test_weight_placeholder)
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
      g = cosine_distances(test_feat, support_feat)
    else:
      raise ValueError("Only cosine similarity is supported.")
    # TODO(rbharath): euclidean kernel is broken!
    #elif self.similarity == 'euclidean':
    #  g = model_ops.euclidean_distance(test_feat, support_feat)
    # Note that gram matrix g has shape (n_test, n_support)

    # soft corresponds to a(xhat, x_i) in eqn (1) of Matching Networks paper
    # https://arxiv.org/pdf/1606.04080v1.pdf
    # Computes softmax across axis 1, (so sums distances to support set for
    # each test entry) to get attention vector
    # Shape (n_test, n_support)
    attention = tf.nn.softmax(g)  # Renormalize

    # Weighted sum of support labels
    # Shape (n_support, 1)
    support_labels = tf.expand_dims(self.support_label_placeholder, 1)
    # pred is yhat in eqn (1) of Matching Networks.
    # Shape squeeze((n_test, n_support) * (n_support, 1)) = (n_test,)
    pred = tf.squeeze(tf.matmul(attention, support_labels), [1])

    # Clip softmax probabilities to range [epsilon, 1-epsilon]
    # Shape (n_test,)
    pred = tf.clip_by_value(pred, 1e-7, 1. - 1e-7)

    # Convert to logit space using inverse sigmoid (logit) function
    # logit function: log(pred) - log(1-pred)
    # Used to invoke tf.nn.sigmoid_cross_entropy_with_logits
    # in Cross Entropy calculation.
    # Shape (n_test,)
    scores = tf.log(pred) - tf.log(tf.constant(1., dtype=tf.float32) - pred)

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
    Parameters
    ----------
    support: dc.data.Dataset
      The support dataset
    test: dc.data.Dataset
      The test dataset
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
    X, y, w, ids = pad_batch(self.test_batch_size, test_batch.X, test_batch.y,
                             test_batch.w, test_batch.ids)
    padded_test_batch = NumpyDataset(X, y, w, ids)
    feed_dict = self.construct_feed_dict(padded_test_batch, support)
    # Get scores
    pred, scores = self.sess.run(
        [self.pred_op, self.scores_op], feed_dict=feed_dict)
    y_pred_batch = np.round(pred)
    # Remove padded elements
    y_pred_batch = y_pred_batch[:n_samples]
    return y_pred_batch

  def predict_proba_on_batch(self, support, test_batch):
    """Make predictions on batch of data."""
    n_samples = len(test_batch)
    X, y, w, ids = pad_batch(self.test_batch_size, test_batch.X, test_batch.y,
                             test_batch.w, test_batch.ids)
    padded_test_batch = NumpyDataset(X, y, w, ids)
    feed_dict = self.construct_feed_dict(padded_test_batch, support)
    # Get scores
    pred, scores = self.sess.run(
        [self.pred_op, self.scores_op], feed_dict=feed_dict)
    # pred corresponds to prob(example == 1)
    y_pred_batch = np.zeros((n_samples, 2))
    # Remove padded elements
    pred = pred[:n_samples]
    y_pred_batch[:, 1] = pred
    y_pred_batch[:, 0] = 1 - pred
    return y_pred_batch

  def evaluate(self,
               dataset,
               metric,
               n_pos,
               n_neg,
               n_trials=1000,
               exclude_support=True):
    """Evaluate performance on dataset according to metrics


    Evaluates the performance of the trained model by sampling
    supports randomly for each task in dataset. For each sampled
    support, the accuracy of the model with support provided is
    computed on all data for that task. If exclude_support is True (by
    default), the support set is excluded from this accuracy
    calculation. exclude_support should be set to false if model's
    memorization capacity wants to be evaluated. 
    

    Since the accuracy on a task is dependent on the choice of random
    support, the evaluation experiment is repeated n_trials times for
    each task.  (Each task gets n_trials experiments). The computed
    accuracies are averaged across trials.

    TODO(rbharath): Currently does not support any transformers.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to test on.
    metrics: dc.metrics.Metric
      Evaluation metric.
    n_pos: int, optional
      Number of positive samples per support.
    n_neg: int, optional
      Number of negative samples per support.
    exclude_support: bool, optional
      Whether support set should be excluded when computing model accuracy.
    """
    # Get batches
    test_tasks = range(len(dataset.get_task_names()))
    task_scores = {task: [] for task in test_tasks}
    support_generator = SupportGenerator(dataset, n_pos, n_neg, n_trials)
    for ind, (task, support) in enumerate(support_generator):
      print("Eval sample %d from task %s" % (ind, str(task)))
      # TODO(rbharath): Add test for get_task_dataset_minus_support for
      # multitask case with missing data...
      if exclude_support:
        print("Removing support datapoints for eval.")
        task_dataset = get_task_dataset_minus_support(dataset, support, task)
      else:
        print("Keeping support datapoints for eval.")
        task_dataset = get_task_dataset(dataset, task)
      y_pred = self.predict_proba(support, task_dataset)
      task_scores[task].append(
          metric.compute_metric(task_dataset.y, y_pred, task_dataset.w))

    # Join information for all tasks.
    mean_task_scores = {}
    std_task_scores = {}
    for task in test_tasks:
      mean_task_scores[task] = np.mean(np.array(task_scores[task]))
      std_task_scores[task] = np.std(np.array(task_scores[task]))
    return mean_task_scores, std_task_scores
