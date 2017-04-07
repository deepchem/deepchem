import threading

import os
import pickle
import time

import networkx as nx
import tensorflow as tf
import numpy as np

from deepchem.data import NumpyDataset
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.models import Model
from deepchem.models.tensorgraph.layers import InputFifoQueue
from deepchem.trans import undo_transforms


class TensorGraph(Model):

  def __init__(self,
               tensorboard=False,
               tensorboard_log_frequency=100,
               learning_rate=0.001,
               batch_size=100,
               use_queue=True,
               mode="classification",
               **kwargs):
    """
    TODO(LESWING) allow a model to change its learning rate
    Parameters
    ----------
    tensorboard: bool
      Should we log to model_dir data for tensorboard?
    learning_rate: float
      learning rate for the model
    kwargs
    """

    # Layer Management
    self.nxgraph = nx.DiGraph()
    self.layers = dict()
    self.parents = dict()
    self.features = list()
    self.labels = list()
    self.outputs = list()
    self.task_weights = list()
    self.loss = None
    self.built = False

    # Singular place to hold Tensor objects which don't serialize
    # These have to be reconstructed on restoring from pickle
    # See TensorGraph._get_tf() for more details on lazy construction
    self.tensor_objects = {
        "FileWriter": None,
        "Graph": tf.Graph(),
        "train_op": None,
        "summary_op": None,
    }
    self.tensorboard = tensorboard
    self.tensorboard_log_frequency = tensorboard_log_frequency
    self.mode = mode
    self.global_step = 0
    self.last_checkpoint = None
    self.input_queue = None
    self.use_queue = use_queue

    self.learning_rate = learning_rate
    self.batch_size = batch_size
    super().__init__(**kwargs)
    self.save_file = "%s/%s" % (self.model_dir, "model")

  def add_layer(self, layer, parents=list()):
    if layer.name in self.layers:
      raise ValueError("Cannot add a layer twice")
    self.nxgraph.add_node(layer.name)
    self.layers[layer.name] = layer
    for parent in parents:
      self.nxgraph.add_edge(parent.name, layer.name)
    self.parents[layer.name] = parents

  def _add_parent(self, layer, parent):
    self.nxgraph.add_edge(parent.name, layer.name)
    self.parents[layer.name].append(parent)

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          checkpoint_interval=10):
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      time1 = time.time()
      train_op = self._get_tf('train_op')
      saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      with tf.Session() as sess:
        self._initialize_weights(sess, saver)
        avg_loss, n_batches = 0.0, 0.0
        for epoch in range(nb_epoch):
          coord = tf.train.Coordinator()
          n_samples = 0
          enqueue_thread = threading.Thread(
              target=_enqueue_batch,
              args=(self, dataset, self._get_tf("Graph"), sess, coord))
          enqueue_thread.start()
          while not coord.should_stop() or n_samples < coord.num_samples:
            output_tensors = [x.out_tensor for x in self.outputs]
            fetches = output_tensors + [train_op, self.loss.out_tensor]
            fetched_values = sess.run(fetches)
            loss = fetched_values[-1]
            avg_loss += loss
            n_batches += 1
            self.global_step += 1
            n_samples += 1
          if epoch % checkpoint_interval == checkpoint_interval - 1:
            saver.save(sess, self.save_file, global_step=self.global_step)
            avg_loss = float(avg_loss) / n_batches
            print('Ending epoch %d: Average loss %g' % (epoch, avg_loss))
        saver.save(sess, self.save_file, global_step=self.global_step)
        self.last_checkpoint = saver.last_checkpoints[-1]
      ############################################################## TIMING
      time2 = time.time()
      print("TIMING: model fitting took %0.3f s" % (time2 - time1))
      ############################################################## TIMING

  def _log_tensorboard(self, sess, feed_dict):
    """
    TODO(LESWING) set epoch
    Parameters
    ----------
    sess
    feed_dict

    Returns
    -------

    """
    if not self.tensorboard:
      return
    summary = sess.run(self._get_tf("summary_op"), feed_dict=feed_dict)
    writer = self._get_tf("FileWriter")
    writer.reopen()
    writer.add_summary(summary, global_step=self.global_step)
    writer.close()

  def fit_on_batch(self, X, y, w):
    if not self.built:
      self.build()
    dataset = NumpyDataset(X, y)
    return self.fit(dataset, nb_epoch=1)

  def _construct_feed_dict(self, X_b, y_b, w_b, ids_b):
    feed_dict = dict()
    if len(self.labels) > 0 and y_b is not None:
      feed_dict[self.labels[0].out_tensor] = y_b
    if len(self.features) > 0 and X_b is not None:
      feed_dict[self.features[0].out_tensor] = X_b
    return feed_dict

  def predict_on_batch(self, X, sess=None):
    """Generates output predictions for the input samples,
      processing the samples in a batched way.

    # Arguments
        x: the input data, as a Numpy array.
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    # Returns
        A Numpy array of predictions.
    """
    retval = self.predict_proba_on_batch(X, sess)
    if self.mode == 'classification':
      return from_one_hot(retval, axis=2)
    return retval

  def predict_proba_on_batch(self, X, sess=None):

    def predict():
      out_tensors = [x.out_tensor for x in self.outputs]
      fetches = out_tensors
      feed_dict = self._construct_feed_dict(X, None, None, None)
      fetched_values = sess.run(fetches, feed_dict=feed_dict)
      return np.array(fetched_values)

    if not self.built:
      self.build()
    if sess is None:
      saver = tf.train.Saver()
      with tf.Session() as sess:
        saver.restore(sess, self.last_checkpoint)
        with self._get_tf("Graph").as_default():
          retval = predict()
    else:
      retval = predict()
    if self.mode == 'classification':  # sample, task, class
      retval = np.transpose(retval, axes=[1, 0, 2])
    elif self.mode == 'regression':  # sample, task
      retval = np.transpose(retval, axes=[1, 0])
    return retval

  def predict(self, dataset, transformers=[], batch_size=None):
    """
    Uses self to make predictions on provided Dataset object.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    if not self.built:
      self.build()
    if batch_size is None:
      batch_size = self.batch_size
    with self._get_tf("Graph").as_default():
      saver = tf.train.Saver()
      with tf.Session() as sess:
        saver.restore(sess, self.last_checkpoint)
        y_preds = []
        n_tasks = self.get_num_tasks()
        for (X_batch, y_b, w_b, ids_batch) in dataset.iterbatches(
            batch_size, deterministic=True):
          y_pred_batch = self.predict_on_batch(X_batch, sess=sess)
          y_pred_batch = undo_transforms(y_pred_batch, transformers)
          y_preds.append(y_pred_batch)
        y_pred = np.vstack(y_preds)

        # The iterbatches does padding with zero-weight examples on the last batch.
        # Remove padded examples.
        n_samples = len(dataset)
        y_pred = y_pred[:n_samples]
        y_pred = np.reshape(y_pred, (n_samples, n_tasks))
        return y_pred

  def predict_proba(self, dataset, transformers=[], batch_size=None):
    """
    TODO: Do transformers even make sense here?

    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    if not self.built:
      self.build()
    if batch_size is None:
      batch_size = self.batch_size
    with self._get_tf("Graph").as_default():
      saver = tf.train.Saver()
      with tf.Session() as sess:
        saver.restore(sess, self.last_checkpoint)
        y_preds = []
        n_tasks = self.get_num_tasks()
        for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(
            batch_size, deterministic=True):
          n_samples = len(X_batch)
          y_pred_batch = self.predict_proba_on_batch(X_batch, sess=sess)
          y_pred_batch = y_pred_batch[:n_samples]
          y_pred_batch = undo_transforms(y_pred_batch, transformers)
          y_preds.append(y_pred_batch)
        y_pred = np.vstack(y_preds)
        # The iterbatches does padding with zero-weight examples on the last batch.
        # Remove padded examples.
        n_samples = len(dataset)
        y_pred = y_pred[:n_samples]
        return y_pred

  def topsort(self):
    return nx.topological_sort(self.nxgraph)

  def build(self):
    with self._get_tf("Graph").as_default():
      self._install_queue()
      order = self.topsort()
      print(order)
      for node in order:
        node_layer = self.layers[node]
        parents = self.parents[node]
        with tf.name_scope(node):
          node_layer.__call__(*parents)
      self.built = True
      if self.use_queue:
        self.input_queue.out_tensors = None

    for layer in self.layers.values():
      if layer.tensorboard:
        self.tensorboard = True
    tf.summary.scalar("loss", self.loss.out_tensor)
    for layer in self.layers.values():
      if layer.tensorboard:
        tf.summary.tensor_summary(layer.name, layer.out_tensor)
    if self.tensorboard:
      writer = self._get_tf("FileWriter")
      writer.add_graph(self._get_tf("Graph"))
      writer.close()

  def _install_queue(self):
    if not self.use_queue:
      for layer in self.features + self.labels + self.task_weights:
        layer.pre_queue = True
      return
    names = []
    shapes = []
    pre_q_inputs = []
    for layer in self.features + self.labels + self.task_weights:
      pre_q_input = layer.create_pre_q(self.batch_size)
      shapes.append(pre_q_input.shape)
      names.append(pre_q_input.name)

      self.add_layer(pre_q_input)
      pre_q_inputs.append(pre_q_input)

    q = InputFifoQueue(shapes, names)
    self.add_layer(q, pre_q_inputs)
    for layer in self.features + self.labels + self.task_weights:
      self._add_parent(layer, q)
    self.input_queue = q

  def set_loss(self, layer):
    self.loss = layer

  def add_label(self, layer):
    self.add_layer(layer)
    self.labels.append(layer)

  def add_feature(self, layer):
    self.add_layer(layer)
    self.features.append(layer)

  def add_output(self, layer):
    self.outputs.append(layer)

  def add_task_weight(self, layer):
    self.add_layer(layer)
    self.task_weights.append(layer)

  def save(self):
    # Remove out_tensor from the object to be pickled
    must_restore = False
    tensor_objects = self.tensor_objects
    self.tensor_objects = {}
    out_tensors = []
    if self.built:
      must_restore = True
      for node in self.topsort():
        node_layer = self.layers[node]
        out_tensors.append(node_layer.none_tensors())
      self.built = False

    # Pickle itself
    pickle_name = os.path.join(self.model_dir, "model.pickle")
    with open(pickle_name, 'wb') as fout:
      pickle.dump(self, fout)

    # add out_tensor back to everyone
    if must_restore:
      for index, node in enumerate(self.topsort()):
        node_layer = self.layers[node]
        node_layer.set_tensors(out_tensors[index])
      self.built = True
    self.tensor_objects = tensor_objects

  def _get_tf(self, obj):
    """
    TODO(LESWING) REALLY NEED TO DOCUMENT THIS
    Parameters
    ----------
    obj

    Returns
    -------
    TensorFlow Object

    """

    if obj in self.tensor_objects and self.tensor_objects[obj] is not None:
      return self.tensor_objects[obj]
    if obj == "Graph":
      self.tensor_objects['Graph'] = tf.Graph()
    elif obj == "FileWriter":
      self.tensor_objects['FileWriter'] = tf.summary.FileWriter(self.model_dir)
    elif obj == 'train_op':
      self.tensor_objects['train_op'] = tf.train.AdamOptimizer(
          self.learning_rate, beta1=.9,
          beta2=.999).minimize(self.loss.out_tensor)
    elif obj == 'summary_op':
      self.tensor_objects['summary_op'] = tf.summary.merge_all(
          key=tf.GraphKeys.SUMMARIES)
    return self._get_tf(obj)

  def _initialize_weights(self, sess, saver):
    """
    Parameters
    ----------
    sess: tf.Session
      The Session must be open
    saver: tf.train.Saver
      A saver object to save/restore checkpoints

    Returns
    -------

    """
    if self.last_checkpoint is None:
      sess.run(tf.global_variables_initializer())
      saver.save(sess, self.save_file, global_step=self.global_step)
    else:
      saver.restore(sess, self.last_checkpoint)

  def get_num_tasks(self):
    return len(self.labels)

  def get_pre_q_input(self, input_layer):
    layer_name = input_layer.name
    pre_q_name = "%s_pre_q" % layer_name
    return self.layers[pre_q_name]

  @staticmethod
  def load_from_dir(model_dir):
    pickle_name = os.path.join(model_dir, "model.pickle")
    with open(pickle_name, 'rb') as fout:
      tensorgraph = pickle.load(fout)
      return tensorgraph


def _enqueue_batch(tg, dataset, graph, sess, coord):
  """
  Function to load data into 
  Parameters
  ----------
  tg
  dataset
  graph
  sess
  coord

  Returns
  -------

  """
  with graph.as_default():
    num_samples = 0
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(
        dataset.iterbatches(tg.batch_size, pad_batches=True)):
      feed_dict = tg._construct_feed_dict(X_b, y_b, w_b, ids_b)
      enq = {}
      for layer in tg.features + tg.labels + tg.task_weights:
        enq[tg.get_pre_q_input(layer).out_tensor] = feed_dict[layer.out_tensor]
      sess.run(tg.input_queue.out_tensor, feed_dict=enq)
      num_samples += 1
      if tg.tensorboard and num_samples % tg.tensorboard_log_frequency == 0:
        tg._log_tensorboard(sess, feed_dict)
    coord.num_samples = num_samples
    coord.request_stop()


class MultiTaskTensorGraph(TensorGraph):
  """
  Class created for legacy sake
  Assumes y is a vector of booleans representing
  classification metrics
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def _construct_feed_dict(self, X_b, y_b, w_b, ids_b):
    feed_dict = dict()
    if y_b is not None:
      for index, label in enumerate(self.labels):
        feed_dict[label.out_tensor] = to_one_hot(y_b[:, index])
    if self.task_weights is not None and w_b is not None:
      feed_dict[self.task_weights.out_tensor] = w_b
    if self.features is not None:
      feed_dict[self.features[0].out_tensor] = X_b
    return feed_dict
