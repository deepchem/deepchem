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
from deepchem.models.tensorgraph.optimizers import Adam
from deepchem.trans import undo_transforms
from deepchem.utils.evaluate import GeneratorEvaluator
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.data.data_loader import featurize_smiles_np


class TensorGraph(Model):

  def __init__(self,
               tensorboard=False,
               tensorboard_log_frequency=100,
               batch_size=100,
               random_seed=None,
               use_queue=True,
               graph=None,
               learning_rate=0.001,
               **kwargs):
    """
    TODO(LESWING) allow a model to change its learning rate
    Parameters
    ----------
    tensorboard: bool
      Should we log to model_dir data for tensorboard?
    tensorboard_log_frequency: int
      How many training batches before logging tensorboard?
    batch_size: int
      default batch size for training and evaluating
    use_queue: boolean
      if True when building we will create a tf.FIFO queue, which will hold
      all features, weights, and labels.  We will feed the inputs into this
      queue in batches of self.batch_size in a separate thread from the
      thread training the model.  You cannot use a queue when
      batches are not of consistent size
    graph: tensorflow.Graph
      the Graph in which to create Tensorflow objects.  If None, a new Graph
      is created.
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
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
    self.optimizer = Adam(
        learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7)

    # Singular place to hold Tensor objects which don't serialize
    # These have to be reconstructed on restoring from pickle
    # See TensorGraph._get_tf() for more details on lazy construction
    self.tensor_objects = {
        "FileWriter": None,
        "Graph": graph,
        "train_op": None,
        "summary_op": None,
    }
    self.tensorboard = tensorboard
    self.tensorboard_log_frequency = tensorboard_log_frequency
    self.tensorboard_step = 0
    #self.mode = mode
    self.global_step = 0
    self.last_checkpoint = None
    self.use_queue = use_queue

    self.batch_size = batch_size
    self.random_seed = random_seed
    super(TensorGraph, self).__init__(**kwargs)
    self.save_file = "%s/%s" % (self.model_dir, "model")
    self.model_class = None

    self.rnn_initial_states = []
    self.rnn_final_states = []
    self.rnn_zero_states = []

  def _add_layer(self, layer):
    if layer.name is None:
      layer.name = "%s_%s" % (layer.__class__.__name__, len(self.layers) + 1)
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
      self._add_layer(in_layer)
      self.nxgraph.add_edge(in_layer.name, layer.name)

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
          yield {self._training_placeholder: 1.0}
      for d in feed_dict_generator:
        feed_dict = {k.out_tensor: v for k, v in six.iteritems(d)}
        feed_dict[self._training_placeholder] = 1.0
        yield feed_dict

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
            self.last_checkpoint = saver.last_checkpoints[-1]
            avg_loss = float(avg_loss) / n_batches
            print('Ending global_step %d: Average loss %g' % (self.global_step,
                                                              avg_loss))
            avg_loss, n_batches = 0.0, 0.0
        avg_loss = float(avg_loss) / n_batches
        print('Ending global_step %d: Average loss %g' % (self.global_step,
                                                          avg_loss))
        saver.save(sess, self.save_file, global_step=self.global_step)
        self.last_checkpoint = saver.last_checkpoints[-1]
      time2 = time.time()
      print("TIMING: model fitting took %0.3f s" % (time2 - time1))

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
        for (initial_state, zero_state) in zip(self.rnn_initial_states,
                                               self.rnn_zero_states):
          feed_dict[initial_state] = zero_state
        yield feed_dict

#  def predict_on_generator(self, generator, transformers=[], outputs=None):
#    """Generates output predictions for the input samples,
#      processing the samples in a batched way.
#
#    Parameters
#    ----------
#    generator: Generator 
#      Generator that constructs feed dictionaries for TensorGraph. 
#    transformers: list
#      List of dc.trans.Transformers.
#    outputs: object 
#      If outputs is None, then will assume outputs = self.outputs[0] (single
#      output). If outputs is a Layer/Tensor, then will evaluate and return as a
#      single ndarray. If outputs is a list of Layers/Tensors, will return a list
#      of ndarrays.
#
#    Returns
#    -------
#    A Numpy array of predictions.
#    """
#    retval = self.predict_proba_on_generator(generator, transformers)
#    if self.mode == 'classification':
#      retval = np.expand_dims(from_one_hot(retval, axis=2), axis=1)
#    return retval

  #def predict_proba_on_generator(self, generator, transformers=[]):
  def predict_on_generator(self, generator, transformers=[], outputs=None):
    """
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    if not self.built:
      self.build()
    if outputs is None:
      assert len(self.outputs) == 1
      outputs = self.outputs
    with self._get_tf("Graph").as_default():
      with tf.Session() as sess:
        saver = tf.train.Saver()
        self._initialize_weights(sess, saver)
        out_tensors = [x.out_tensor for x in self.outputs]
        results = []
        for feed_dict in generator:
          feed_dict = {
              self.layers[k.name].out_tensor: v
              for k, v in six.iteritems(feed_dict)
          }
          feed_dict[self._training_placeholder] = 0.0
          result = sess.run(out_tensors, feed_dict=feed_dict)
          result = undo_transforms(result, transformers)
          results.append(result)
        if len(results) == 1:
          return results[0]
        else:
          return results
        #return np.concatenate(results, axis=0)

#  def bayesian_predict_on_batch(self, X, transformers=[], n_passes=4):
#    """
#    Returns:
#      mu: numpy ndarray of shape (n_samples, n_tasks)
#      sigma: numpy ndarray of shape (n_samples, n_tasks)
#    """
#    dataset = NumpyDataset(X=X, y=None, n_tasks=len(self.outputs))
#    y_ = []
#    for i in range(n_passes):
#      generator = self.default_generator(
#          dataset, predict=True, pad_batches=True)
#      y_.append(self.predict_on_generator(generator, transformers))
#
#    y_ = np.concatenate(y_, axis=2)
#    mu = np.mean(y_, axis=2)
#    sigma = np.std(y_, axis=2)
#
#    return mu, sigma

#  def predict_on_smiles_batch(self,
#                              smiles,
#                              featurizer,
#                              n_tasks,
#                              transformers=[]):
#    """
#    # Returns:
#      A numpy ndarray of shape (n_samples, n_tasks)
#    """
#    convmols = featurize_smiles_np(smiles, featurizer)
#
#    dataset = NumpyDataset(X=convmols, y=None, n_tasks=len(self.outputs))
#    generator = self.default_generator(dataset, predict=True, pad_batches=True)
#    return self.predict_on_generator(generator, transformers)

#  def predict_on_batch(self, X, sess=None, transformers=[]):
#    """Generates output predictions for the input samples,
#      processing the samples in a batched way.
#
#    # Arguments
#        x: the input data, as a Numpy array.
#        verbose: verbosity mode, 0 or 1.
#
#    # Returns
#        A Numpy array of predictions.
#    """
#    dataset = NumpyDataset(X=X, y=None)
#    generator = self.default_generator(dataset, predict=True, pad_batches=False)
#    return self.predict_on_generator(generator, transformers)

#  def predict_proba_on_batch(self, X, sess=None, transformers=[]):
#    dataset = NumpyDataset(X=X, y=None)
#    generator = self.default_generator(dataset, predict=True, pad_batches=False)
#    return self.predict_proba_on_generator(generator, transformers)

  def predict(self, dataset, transformers=[], outputs=None):
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list
      List of dc.trans.Transformers.
    outputs: object 
      If outputs is None, then will assume outputs = self.outputs[0] (single
      output). If outputs is a Layer/Tensor, then will evaluate and return as a
      single ndarray. If outputs is a list of Layers/Tensors, will return a list
      of ndarrays.

    Returns
    -------
    results: numpy ndarray or list of numpy ndarrays
    """
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers, outputs)

#  def predict_proba(self, dataset, transformers=[], outputs=None):
#    """
#    Parameters
#    ----------
#    dataset: dc.data.Dataset
#      Dataset to make prediction on
#    transformers: list
#      List of dc.trans.Transformers.
#    outputs: object 
#      If outputs is None, then will assume outputs = self.outputs[0] (single
#      output). If outputs is a Layer/Tensor, then will evaluate and return as a
#      single ndarray. If outputs is a list of Layers/Tensors, will return a list
#      of ndarrays.
#
#    Returns
#    -------
#    y_pred: numpy ndarray or list of numpy ndarrays
#    """
#    generator = self.default_generator(dataset, predict=True, pad_batches=False)
#    return self.predict_proba_on_generator(generator, transformers, output)

  def topsort(self):
    return nx.topological_sort(self.nxgraph)

  def build(self):
    if self.built:
      return
    with self._get_tf("Graph").as_default():
      self._training_placeholder = tf.placeholder(dtype=tf.float32, shape=())
      if self.random_seed is not None:
        tf.set_random_seed(self.random_seed)
      self._install_queue()
      order = self.topsort()
      for node in order:
        with tf.name_scope(node):
          node_layer = self.layers[node]
          node_layer.create_tensor(training=self._training_placeholder)
          self.rnn_initial_states += node_layer.rnn_initial_states
          self.rnn_final_states += node_layer.rnn_final_states
          self.rnn_zero_states += node_layer.rnn_zero_states
          node_layer.add_summary_to_tg()

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

    # As a sanity check, make sure all tensors have the correct shape.

    for layer in self.layers.values():
      try:
        assert list(layer.shape) == layer.out_tensor.get_shape().as_list(
        ), '%s: Expected shape %s does not match actual shape %s' % (
            layer.name, layer.shape, layer.out_tensor.get_shape().as_list())
      except NotImplementedError:
        pass

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
    q.name = "%s_%s" % (q.__class__.__name__, len(self.layers) + 1)

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

  def set_optimizer(self, optimizer):
    """Set the optimizer to use for fitting."""
    self.optimizer = optimizer

  def get_pickling_errors(self, obj, seen=None):
    if seen == None:
      seen = []
    try:
      state = obj.__getstate__()
    except AttributeError:
      return
    if state == None:
      return
    if isinstance(state, tuple):
      if not isinstance(state[0], dict):
        state = state[1]
      else:
        state = state[0].update(state[1])
    result = {}
    for i in state:
      try:
        pickle.dumps(state[i], protocol=2)
      except pickle.PicklingError:
        if not state[i] in seen:
          seen.append(state[i])
          result[i] = self.get_pickling_errors(state[i], seen)
    return result

  def save(self):
    # Remove out_tensor from the object to be pickled
    must_restore = False
    tensor_objects = self.tensor_objects
    rnn_initial_states = self.rnn_initial_states
    rnn_final_states = self.rnn_final_states
    rnn_zero_states = self.rnn_zero_states
    self.tensor_objects = {}
    self.rnn_initial_states = []
    self.rnn_final_states = []
    self.rnn_zero_states = []
    out_tensors = []
    if self.built:
      must_restore = True
      for node in self.topsort():
        node_layer = self.layers[node]
        out_tensors.append(node_layer.none_tensors())
      optimizer = self.optimizer
      self.optimizer = None
      training_placeholder = self._training_placeholder
      self._training_placeholder = None
      self.built = False

    # Pickle itself
    pickle_name = os.path.join(self.model_dir, "model.pickle")
    with open(pickle_name, 'wb') as fout:
      try:
        pickle.dump(self, fout)
      except Exception as e:
        print(self.get_pickling_errors(self))
        raise e

    # add out_tensor back to everyone
    if must_restore:
      for index, node in enumerate(self.topsort()):
        node_layer = self.layers[node]
        node_layer.set_tensors(out_tensors[index])
      self._training_placeholder = training_placeholder
      self.optimizer = optimizer
      self.built = True
    self.tensor_objects = tensor_objects
    self.rnn_initial_states = rnn_initial_states
    self.rnn_final_states = rnn_final_states
    self.rnn_zero_states = rnn_zero_states

  def evaluate_generator(self,
                         feed_dict_generator,
                         metrics,
                         transformers=[],
                         labels=None,
                         outputs=None,
                         weights=[],
                         per_task_metrics=False):

    if labels is None:
      raise ValueError
    n_tasks = len(self.outputs)
    n_classes = self.outputs[0].out_tensor.get_shape()[-1].value
    evaluator = GeneratorEvaluator(
        self,
        feed_dict_generator,
        transformers,
        labels=labels,
        outputs=outputs,
        weights=weights,
        n_tasks=n_tasks,
        n_classes=n_classes)
    if not per_task_metrics:
      scores = evaluator.compute_model_performance(metrics)
      return scores
    else:
      scores, per_task_scores = evaluator.compute_model_performance(
          metrics, per_task_metrics=per_task_metrics)
      return scores, per_task_scores

  def get_layer_variables(self, layer):
    """Get the list of trainable variables in a layer of the graph."""
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      return tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=layer.variable_scope)

  def get_global_step(self):
    return self._get_tf("GlobalStep")

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
    elif obj == 'Optimizer':
      self.tensor_objects['Optimizer'] = self.optimizer._create_optimizer(
          self._get_tf('GlobalStep'))
    elif obj == 'train_op':
      self.tensor_objects['train_op'] = self._get_tf('Optimizer').minimize(
          self.loss.out_tensor, global_step=self._get_tf('GlobalStep'))
    elif obj == 'summary_op':
      self.tensor_objects['summary_op'] = tf.summary.merge_all(
          key=tf.GraphKeys.SUMMARIES)
    elif obj == 'GlobalStep':
      with self._get_tf("Graph").as_default():
        self.tensor_objects['GlobalStep'] = tf.Variable(0, trainable=False)
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
      self.last_checkpoint = saver.last_checkpoints[-1]
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

  def __del__(self):
    pass


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
      enq[tg._training_placeholder] = 1.0
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


class TFWrapper(object):
  """This class exists as a workaround for Tensorflow objects not being picklable.

  The job of a TFWrapper is to create Tensorflow objects by passing defined arguments
  to a constructor.  There are cases where we really want to store Tensorflow objects
  of various sorts (optimizers, initializers, etc.), but we can't because they cannot
  be pickled.  So instead we store a TFWrapper that creates the object when needed.
  """

  def __init__(self, tf_class, **kwargs):
    """Create a TFWrapper for constructing a Tensorflow object.

    Parameters
    ----------
    tf_class: class
      the type of object to create
    kwargs:
      any other arguments will be passed on to the object's constructor
    """
    self.tf_class = tf_class
    self.kwargs = kwargs

  def __call__(self):
    return self.tf_class(**self.kwargs)
