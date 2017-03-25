import os
import pickle
import time

import networkx as nx
import tensorflow as tf
import numpy as np

from deepchem.data import NumpyDataset
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.models import Model


class TensorGraph(Model):
  def __init__(self, tensorboard=False, learning_rate=0.001, **kwargs):
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
    self.epoch = 0
    self.last_checkpoint = None
    self.tensorboard = tensorboard

    self.learning_rate = learning_rate
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

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          batch_size=50,
          checkpoint_interval=10):
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      time1 = time.time()
      print("Training for %d epochs" % nb_epoch)
      train_op = self._get_tf('train_op')
      saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      with tf.Session() as sess:
        self._initialize_weights(sess, saver)
        for self.epoch in range(self.epoch, self.epoch + nb_epoch):
          avg_loss, n_batches = 0., 0
          for ind, (X_b, y_b, w_b, ids_b) in enumerate(
            dataset.iterbatches(batch_size, pad_batches=True)):
            if ind % log_every_N_batches == 0:
              print("On batch %d" % ind)
            feed_dict = self._construct_feed_dict(X_b, y_b, w_b, ids_b)
            output_tensors = [x.out_tensor for x in self.outputs]
            fetches = output_tensors + [train_op, self.loss.out_tensor]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            loss = fetched_values[-1]
            avg_loss += loss
            n_batches += 1
            self._log_tensorboard(sess, feed_dict)
          if self.epoch % checkpoint_interval == checkpoint_interval - 1:
            saver.save(sess, self.save_file, global_step=self.epoch)
          avg_loss = float(avg_loss) / n_batches
          print('Ending epoch %d: Average loss %g' % (self.epoch, avg_loss))
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
    print("Loggin")
    writer = self._get_tf("FileWriter")
    writer.reopen()
    writer.add_summary(summary, current_global_step=self.epoch)
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

  def predict_on_batch(self, X):
    """Generates output predictions for the input samples,
      processing the samples in a batched way.

    # Arguments
        x: the input data, as a Numpy array.
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    # Returns
        A Numpy array of predictions.
    """
    if len(self.features) != 1:
      raise ValueError("Only allow one input set of features")
    features = self.features[0]
    if not self.built:
      self.build()
    with self.tensor_objects['Graph'].as_default():
      saver = tf.train.Saver()
      with tf.Session() as sess:
        saver.restore(sess, self.last_checkpoint)
        fetches = [x.out_tensor for x in self.outputs]
        feed_dict = {features.out_tensor: X}
        fetched_values = sess.run(fetches, feed_dict=feed_dict)
        return np.array(fetched_values)

  def predict_proba_on_batch(self, X):
    if not self.built:
      self.build()
    with self.tensor_objects['Graph'].as_default():
      saver = tf.train.Saver()
      with tf.Session() as sess:
        saver.restore(sess, self.last_checkpoint)
        out_tensors = [x.out_tensor for x in self.outputs]
        fetches = out_tensors
        feed_dict = self._construct_feed_dict(X, None, None, None)
        fetched_values = sess.run(fetches, feed_dict=feed_dict)
        return np.array(fetched_values)

  def topsort(self):
    return nx.topological_sort(self.nxgraph)

  def build(self):
    with self.tensor_objects['Graph'].as_default():
      order = self.topsort()
      for node in order:
        node_layer = self.layers[node]
        parents = self.parents[node]
        with tf.name_scope(node):
          node_layer.__call__(*parents)
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
      writer.add_graph(self.tensor_objects['Graph'])
      writer.close()

  def set_loss(self, layer):
    self.loss = layer

  def add_label(self, layer):
    self.labels.append(layer)

  def add_feature(self, layer):
    self.features.append(layer)

  def add_output(self, layer):
    self.outputs.append(layer)

  def save(self):
    # Remove out_tensor from the object to be pickled
    must_restore = False
    out_tensors = []
    tensor_objects = self.tensor_objects
    self.tensor_objects = {}
    if self.built:
      must_restore = True
      out_tensors = []
      for node in self.topsort():
        node_layer = self.layers[node]
        out_tensors.append(node_layer.out_tensor)
        node_layer.out_tensor = None
      self.built = False

    # Pickle itself
    pickle_name = os.path.join(self.model_dir, "model.pickle")
    with open(pickle_name, 'wb') as fout:
      pickle.dump(self, fout)

    # add out_tensor back to everyone
    if must_restore:
      for index, node in enumerate(self.topsort()):
        node_layer = self.layers[node]
        node_layer.out_tensor = out_tensors[index]
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

    if self.tensor_objects[obj] is not None:
      return self.tensor_objects[obj]
    if obj == "Graph":
      self.tensor_objects['Graph'] = tf.Graph()
    elif obj == "FileWriter":
      self.tensor_objects['FileWriter'] = tf.summary.FileWriter(self.model_dir)
    elif obj == 'train_op':
      self.tensor_objects['train_op'] = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss.out_tensor)
    elif obj == 'summary_op':
      self.tensor_objects['summary_op'] = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
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
      saver.save(sess, self.save_file, global_step=self.epoch)
    else:
      saver.restore(sess, self.last_checkpoint)


@staticmethod
def load_from_dir(model_dir):
  pickle_name = os.path.join(model_dir, "model.pickle")
  with open(pickle_name, 'rb') as fout:
    tensorgraph = pickle.load(fout)
    tensorgraph.graph = tf.Graph()
    return tensorgraph


class MultiTaskTensorGraph(TensorGraph):
  """
  Class created for legacy sake
  Assumes y is a vector of booleans representing
  classification metrics
  """

  def __init__(self, **kwargs):
    self.task_weights = None
    super().__init__(**kwargs)

  def set_task_weights(self, layer):
    self.task_weights = layer

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

  def get_num_tasks(self):
    return len(self.labels)

  def predict_on_batch(self, X):
    # sample x task
    # Class is implied by the value of task [0,1]
    prediction = super(MultiTaskTensorGraph, self).predict_on_batch(X)
    prediction = np.transpose(from_one_hot(prediction, axis=2))
    return prediction

  def predict_proba_on_batch(self, X):
    prediction = super(MultiTaskTensorGraph, self).predict_on_batch(X)
    # sample x task x class
    prediction1 = np.transpose(prediction, axes=[1, 0, 2])
    return prediction1
