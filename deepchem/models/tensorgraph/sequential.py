"""
Convenience class for building sequential deep networks.
"""
from __future__ import division
from __future__ import unicode_literals

import warnings
import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Label
from deepchem.models.tensorgraph.layers import SoftMaxCrossEntropy
from deepchem.models.tensorgraph.layers import ReduceMean
from deepchem.models.tensorgraph.layers import ReduceSquareDifference
from deepchem.models.tensorgraph.support_layers import GraphTopology


class Sequential(TensorGraph):
  """Sequential models are linear stacks of layers.

  Analogous to the Sequential model from Keras and allows for less
  verbose construction of simple deep learning model.

  Example
  -------

  >>> import deepchem as dc
  >>> import numpy as np
  >>> from deepchem.models.tensorgraph import layers
  >>> # Define Data
  >>> X = np.random.rand(20, 2)                     
  >>> y = [[0, 1] for x in range(20)]
  >>> dataset = dc.data.NumpyDataset(X, y)                              
  >>> model = dc.models.Sequential(learning_rate=0.01)                  
  >>> model.add(layers.Dense(out_channels=2))                                  
  >>> model.add(layers.SoftMax())
  """

  def __init__(self, **kwargs):
    """Initializes a sequential model
    """
    self.num_layers = 0
    self._prev_layer = None
    if "use_queue" in kwargs:
      if kwargs["use_queue"]:
        raise ValueError("Sequential doesn't support queues.")
    kwargs["use_queue"] = False
    self._layer_list = []
    self._built = False
    super(Sequential, self).__init__(**kwargs)

  def add(self, layer):
    """Adds a new layer to model.

    Parameter
    ---------
    layer: Layer
      Adds layer to this graph.
    """
    self._layer_list.append(layer)

  def fit(self, dataset, loss, **kwargs):
    """Fits on the specified dataset.

    If called for the first time, constructs the TensorFlow graph for this
    model. Fits this graph on the specified dataset according to the specified
    loss.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset with data
    loss: string
      Only "binary_crossentropy" or "mse" for now.
    """
    X_shape, y_shape, _, _ = dataset.get_shape()
    # Calling fit() for first time
    if not self.built:
      feature_shape = X_shape[1:]
      label_shape = y_shape[1:]
      # Add in features
      features = Feature(shape=(None,) + feature_shape)
      # Add in labels
      labels = Label(shape=(None,) + label_shape)

      # Add in all layers
      prev_layer = features
      if len(self._layer_list) == 0:
        raise ValueError("No layers have been added to model.")
      for ind, layer in enumerate(self._layer_list):
        if len(layer.in_layers) > 1:
          raise ValueError("Cannot specify more than one "
                           "in_layer for Sequential.")
        layer.in_layers += [prev_layer]
        prev_layer = layer
      # The last layer is the output of the model
      self.outputs.append(prev_layer)

      if loss == "binary_crossentropy":
        smce = SoftMaxCrossEntropy(in_layers=[labels, prev_layer])
        self.set_loss(ReduceMean(in_layers=[smce]))
      elif loss == "mse":
        mse = ReduceSquareDifference(in_layers=[prev_layer, labels])
        self.set_loss(mse)
      else:
        # TODO(rbharath): Add in support for additional
        # losses.
        raise ValueError("Unsupported loss.")

    super(Sequential, self).fit(dataset, **kwargs)

  def restore(self, checkpoint=None):
    """Not currently supported.
    """
    # TODO(rbharath): The TensorGraph can't be built until
    # fit is called since the shapes of features/labels
    # not specified. Need to figure out a good restoration
    # method for this use case.
    raise ValueError("Restore is not yet supported " "for sequential models.")


class SequentialSupport(TensorGraph):
  """Sequential support models are used for training of low data models.

  Sequential support models are used to implement low data models
  that use the concept of "support sets" and "test sets." A "support"
  is a small chunk of evidence. A "test set" contains queries which
  are to be answered conditioned on the support. Sequential support
  graph models are trained by repeatedly sampling supports from
  available data, taking gradient descent steps, and continuing. For
  more details on this training style, see
  https://pubs.acs.org/doi/abs/10.1021/acscentsci.6b00367.

  Technically, these models support two parallel towers. One tower is
  used for processing support sets. The second is used for processing
  the test set. Adding a layer to SequentialSupport adds it to
  both the support tower and the test tower.
  """

  def __init__(self, n_feat, **kwargs):
    """Initializes model with no layers.

    Parameters
    ----------
    n_feat: int
      Number of atomic features.
    """
    # Create graph topology and x
    self.test_graph_topology = GraphTopology(n_feat, name='test')
    self.support_graph_topology = GraphTopology(n_feat, name='support')
    self.test = self.test_graph_topology.get_atom_features_placeholder()
    self.support = self.support_graph_topology.get_atom_features_placeholder()

    # Keep track of the layers
    self.layers = []
    # Whether or not we have used the GraphGather layer yet
    self.bool_pre_gather = True
    super(SequentialSupport, self).__init__(**kwargs)

  def add_placeholders(self):
    """Adds placeholders to graph."""
    #################################################################### DEBUG
    #self.test_label_placeholder = Input(
    #    tensor=tf.placeholder(dtype='float32', shape=(self.test_batch_size),
    #    name="label_placeholder"))
    #self.test_weight_placeholder = Input(
    #    tensor=tf.placeholder(dtype='float32', shape=(self.test_batch_size),
    #    name="weight_placeholder"))
    self.test_label_placeholder = tf.placeholder(
        dtype='float32', shape=(self.test_batch_size), name="label_placeholder")
    self.test_weight_placeholder = tf.placeholder(
        dtype='float32',
        shape=(self.test_batch_size),
        name="weight_placeholder")

    # TODO(rbharath): Should weights for the support be used?
    # Support labels
    #self.support_label_placeholder = Input(
    #    tensor=tf.placeholder(dtype='float32', shape=[self.support_batch_size],
    #    name="support_label_placeholder"))
    self.support_label_placeholder = tf.placeholder(
        dtype='float32',
        shape=[self.support_batch_size],
        name="support_label_placeholder")
    self.phase = tf.placeholder(dtype='bool', name='keras_learning_phase')

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
        self.support = layer(
            [self.support] + self.support_graph_topology.topology)
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
        self.support = layer(
            [self.support] + self.support_graph_topology.topology)
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

  def construct_feed_dict(self, test, support, training=True, add_phase=False):
    """Constructs tensorflow feed from test/support sets."""
    # Generate dictionary elements for support
    feed_dict = (self.model.support_graph_topology.batch_to_feed_dict(
        support.X))
    feed_dict[self.support_label_placeholder] = np.squeeze(support.y)
    # Get graph information for test
    batch_topo_dict = (self.model.test_graph_topology.batch_to_feed_dict(
        test.X))
    feed_dict = merge_dicts([batch_topo_dict, feed_dict])
    # Generate dictionary elements for test
    feed_dict[self.test_label_placeholder] = np.squeeze(test.y)
    feed_dict[self.test_weight_placeholder] = np.squeeze(test.w)

    if add_phase:
      feed_dict[self.phase] = training
    return feed_dict

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
      g = model_ops.cosine_distances(test_feat, support_feat)
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
