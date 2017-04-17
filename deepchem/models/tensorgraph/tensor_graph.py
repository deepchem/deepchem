import pickle
import threading
import time

import networkx as nx
import numpy as np
import os
import six
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from deepchem.data import NumpyDataset
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.models import Model
from deepchem.models.tensorgraph.layers import InputFifoQueue, Label, Feature, Weights
from deepchem.trans import undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator


class TensorGraph(Model):

  def __init__(self,
               tensorboard=False,
               tensorboard_log_frequency=100,
               learning_rate=0.001,
               batch_size=100,
               use_queue=True,
               mode="regression",
               **kwargs):
    """
    TODO(LESWING) allow a model to change its learning rate
    Parameters
    ----------
    tensorboard: bool
      Should we log to model_dir data for tensorboard?
    tensorboard_log_frequency: int
      How many training batches before logging tensorboard?
    learning_rate: float
      learning rate for optimizer
    batch_size: int
      default batch size for training and evaluating
    use_queue: boolean
      if True when building we will create a tf.FIFO queue, which will hold
      all features, weights, and labels.  We will feed the inputs into this
      queue in batches of self.batch_size in a separate thread from the
      thread training the model.  You cannot use a queue when
      batches are not of consistent size
    mode: str
      "regression" or "classification".  "classification" models on
      predict will do an argmax(axis=2) to determine the class of the
      prediction.
    kwargs
    """

    # Layer Management
    self.nxgraph = nx.DiGraph()
    self.layers = dict()
    self.features = list()
    self.labels = list()
    self.outputs = list()
    self.task_weights = list()
    self.loss = None
    self.built = False
    self.queue_installed = False

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
    self.tensorboard_step = 0
    self.mode = mode
    self.global_step = 0
    self.last_checkpoint = None
    self.use_queue = use_queue

    self.learning_rate = learning_rate
    self.batch_size = batch_size
    super(TensorGraph, self).__init__(**kwargs)
    self.save_file = "%s/%s" % (self.model_dir, "model")
    self.model_class = None

  def _add_layer(self, layer):
    if layer.name in self.layers:
      return
    if isinstance(layer, Feature):
      self.features.append(layer)
    if isinstance(layer, Label):
      self.labels.append(layer)
    if isinstance(layer, Weights):
      self.task_weights.append(layer)
    self.nxgraph.add_node(layer.name)
    self.layers[layer.name] = layer
    for in_layer in layer.in_layers:
      self.nxgraph.add_edge(in_layer.name, layer.name)
      self._add_layer(in_layer)

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          checkpoint_interval=1000):
    return self.fit_generator(
        self.default_generator(dataset, epochs=nb_epoch),
        max_checkpoints_to_keep, checkpoint_interval)

  def fit_generator(self,
                    feed_dict_generator,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000):

    def create_feed_dict():
      if self.use_queue:
        while True:
          yield {}
      for d in feed_dict_generator:
        yield {k.out_tensor: v for k, v in six.iteritems(d)}

    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      time1 = time.time()
      train_op = self._get_tf('train_op')
      saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      with tf.Session() as sess:
        self._initialize_weights(sess, saver)
        avg_loss, n_batches = 0.0, 0.0
        coord = tf.train.Coordinator()
        n_samples = 0
        if self.use_queue:
          enqueue_thread = threading.Thread(
              target=_enqueue_batch,
              args=(self, feed_dict_generator, self._get_tf("Graph"), sess,
                    coord))
          enqueue_thread.start()
        output_tensors = [x.out_tensor for x in self.outputs]
        fetches = output_tensors + [train_op, self.loss.out_tensor]
        for feed_dict in create_feed_dict():
          try:
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            loss = fetched_values[-1]
            avg_loss += loss
            n_batches += 1
            self.global_step += 1
            n_samples += 1
            if self.tensorboard and n_samples % self.tensorboard_log_frequency == 0:
              summary = sess.run(
                  self._get_tf("summary_op"), feed_dict=feed_dict)
              self._log_tensorboard(summary)
          except OutOfRangeError:
            break
          if self.global_step % checkpoint_interval == checkpoint_interval - 1:
            saver.save(sess, self.save_file, global_step=self.global_step)
            avg_loss = float(avg_loss) / n_batches
            print('Ending global_step %d: Average loss %g' %
                  (self.global_step, avg_loss))
        avg_loss = float(avg_loss) / n_batches
        print('Ending global_step %d: Average loss %g' %
              (self.global_step, avg_loss))
        saver.save(sess, self.save_file, global_step=self.global_step)
        self.last_checkpoint = saver.last_checkpoints[-1]
      ############################################################## TIMING
      time2 = time.time()
      print("TIMING: model fitting took %0.3f s" % (time2 - time1))
      ############################################################## TIMING

  def _log_tensorboard(self, summary):
    """
    TODO(LESWING) set epoch
    Parameters
    ----------
    Returns
    -------
    """
    global_step = int(self.global_step)
    writer = self._get_tf("FileWriter")
    writer.reopen()
    writer.add_summary(summary, global_step=global_step)
    writer.close()

  def fit_on_batch(self, X, y, w):
    if not self.built:
      self.build()
    dataset = NumpyDataset(X, y)
    return self.fit(dataset, nb_epoch=1)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    if len(self.features) > 1:
      raise ValueError("More than one Feature, must use generator")
    if len(self.labels) > 1:
      raise ValueError("More than one Label, must use generator")
    if len(self.task_weights) > 1:
      raise ValueError("More than one Weights, must use generator")
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):
        feed_dict = dict()
        if len(self.labels) == 1 and y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b
        if len(self.features) == 1 and X_b is not None:
          feed_dict[self.features[0]] = X_b
        if len(self.task_weights) == 1 and w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        yield feed_dict

  def predict_on_generator(self, generator, transformers=[]):
    """Generates output predictions for the input samples,
      processing the samples in a batched way.

    # Arguments
        x: the input data, as a Numpy array.
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    # Returns
        A Numpy array of predictions.
    """
    retval = self.predict_proba_on_generator(generator, transformers)
    if self.mode == 'classification':
      retval = np.expand_dims(from_one_hot(retval, axis=2), axis=1)
    return retval

  def predict_proba_on_generator(self, generator, transformers=[]):
    """
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, self.last_checkpoint)
        out_tensors = [x.out_tensor for x in self.outputs]
        results = []
        for feed_dict in generator:
          feed_dict = {
              self.layers[k.name].out_tensor: v
              for k, v in six.iteritems(feed_dict)
          }
          result = np.array(sess.run(out_tensors, feed_dict=feed_dict))
          if len(result.shape) == 3:
            result = np.transpose(result, axes=[1, 0, 2])
          result = result.squeeze(axis=0)
          result = undo_transforms(result, transformers)
          results.append(result)
        return np.concatenate(results, axis=0)

  def predict_on_batch(self, X, sess=None, transformers=[]):
    """Generates output predictions for the input samples,
      processing the samples in a batched way.

    # Arguments
        x: the input data, as a Numpy array.
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    # Returns
        A Numpy array of predictions.
    """
    dataset = NumpyDataset(X=X, y=None)
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers)

  def predict_proba_on_batch(self, X, sess=None, transformers=[]):
    dataset = NumpyDataset(X=X, y=None)
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_proba_on_generator(generator, transformers)

  def predict(self, dataset, transformers=[], batch_size=None):
    """
    Uses self to make predictions on provided Dataset object.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers)

  def predict_proba(self, dataset, transformers=[], batch_size=None):
    """
    TODO: Do transformers even make sense here?

    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_proba_on_generator(generator, transformers)

  def topsort(self):
    return nx.topological_sort(self.nxgraph)

  def build(self):
    if self.built:
      return
    with self._get_tf("Graph").as_default():
      self._install_queue()
      order = self.topsort()
      print(order)
      for node in order:
        with tf.name_scope(node):
          node_layer = self.layers[node]
          node_layer._create_tensor()
      self.built = True

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
    """
    """
    if not self.use_queue or self.queue_installed:
      for layer in self.features + self.labels + self.task_weights:
        layer.pre_queue = True
      return
    names = []
    shapes = []
    pre_q_inputs = []
    q = InputFifoQueue(shapes, names, in_layers=pre_q_inputs)
    for layer in self.features + self.labels + self.task_weights:
      pre_q_input = layer.create_pre_q(self.batch_size)
      shapes.append(pre_q_input.shape)
      names.append(pre_q_input.name)
      pre_q_inputs.append(pre_q_input)

      layer.in_layers.append(q)
      self.nxgraph.add_edge(q.name, layer.name)

    self._add_layer(q)
    self.input_queue = q
    self.queue_installed = True

  def set_loss(self, layer):
    self._add_layer(layer)
    self.loss = layer

  def add_output(self, layer):
    self._add_layer(layer)
    self.outputs.append(layer)

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

  def evaluate_generator(self,
                         dataset,
                         metrics,
                         transformers=[],
                         labels=None,
                         outputs=None,
                         weights=[],
                         per_task_metrics=False):

    if labels is None:
      raise ValueError
    evaluator = GeneratorEvaluator(
        self,
        dataset,
        transformers,
        labels=labels,
        outputs=outputs,
        weights=weights)
    if not per_task_metrics:
      scores = evaluator.compute_model_performance(metrics)
      return scores
    else:
      scores, per_task_scores = evaluator.compute_model_performance(
          metrics, per_task_metrics=per_task_metrics)
      return scores, per_task_scores

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
    return len(self.outputs)

  def get_pre_q_input(self, input_layer):
    layer_name = input_layer.name
    pre_q_name = "%s_pre_q" % layer_name
    return self.layers[pre_q_name]

  @staticmethod
  def load_from_dir(model_dir):
    pickle_name = os.path.join(model_dir, "model.pickle")
    with open(pickle_name, 'rb') as fout:
      tensorgraph = pickle.load(fout)
      tensorgraph.built = False
      return tensorgraph


def _enqueue_batch(tg, generator, graph, sess, coord):
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
    for feed_dict in generator:
      enq = {}
      for layer in tg.features + tg.labels + tg.task_weights:
        enq[tg.get_pre_q_input(layer).out_tensor] = feed_dict[layer]
      sess.run(tg.input_queue.out_tensor, feed_dict=enq)
      num_samples += 1
      if tg.tensorboard and num_samples % tg.tensorboard_log_frequency == 0:
        enq = {k.out_tensor: v for k, v in six.iteritems(feed_dict)}
        summary = sess.run(tg._get_tf("summary_op"), feed_dict=enq)
        tg._log_tensorboard(summary)
    sess.run(tg.input_queue.close_op)
    coord.num_samples = num_samples
    coord.request_stop()
