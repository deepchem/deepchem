from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import time
import numpy as np
import tensorflow as tf

from deepchem.utils.save import log
from deepchem.models.tensorflow_models import TensorflowGraph
from deepchem.models.tensorflow_models import TensorflowGraphModel
from deepchem.models.tensorflow_models.lr import TensorflowLogisticRegression


class TensorflowMultiTaskIRVClassifier(TensorflowLogisticRegression):

  def __init__(self,
               n_tasks,
               K=10,
               logdir=None,
               n_classes=2,
               penalty=0.0,
               penalty_type="l2",
               learning_rate=0.001,
               momentum=.8,
               optimizer="adam",
               batch_size=50,
               verbose=True,
               seed=None,
               **kwargs):
    """Initialize TensorflowMultiTaskIRVClassifier
    
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    K: int
      Number of nearest neighbours used in classification
    logdir: str
      Location to save data
    n_classes: int
      number of different labels
    penalty: float
      Amount of penalty (l2 or l1 applied)
    penalty_type: str
      Either "l2" or "l1"
    learning_rate: float
      Learning rate for model.
    momentum: float
      Momentum. Only applied if optimizer=="momentum"
    optimizer: str
      Type of optimizer applied.
    batch_size: int
      Size of minibatches for training.
    verbose: True 
      Perform logging.
    seed: int
      If not none, is used as random seed for tensorflow.        

    """
    warnings.warn("The TensorflowMultiTaskIRVClassifier is "
                  "deprecated. Will be removed in DeepChem 1.4.",
                  DeprecationWarning)
    self.n_tasks = n_tasks
    self.K = K
    self.n_features = 2 * self.K * self.n_tasks
    print("n_features after fit_transform: %d" % int(self.n_features))
    TensorflowGraphModel.__init__(
        self,
        n_tasks,
        self.n_features,
        logdir=logdir,
        layer_sizes=None,
        weight_init_stddevs=None,
        bias_init_consts=None,
        penalty=penalty,
        penalty_type=penalty_type,
        dropouts=None,
        n_classes=n_classes,
        learning_rate=learning_rate,
        momentum=momentum,
        optimizer=optimizer,
        batch_size=batch_size,
        pad_batches=False,
        verbose=verbose,
        seed=seed,
        **kwargs)

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture of IRV as described in:
       
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2750043/
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    K = self.K
    with graph.as_default():
      output = []
      with placeholder_scope:
        mol_features = tf.placeholder(
            tf.float32, shape=[None, self.n_features], name='mol_features')
      with tf.name_scope('variable'):
        V = tf.Variable(tf.constant([0.01, 1.]), name="vote", dtype=tf.float32)
        W = tf.Variable(tf.constant([1., 1.]), name="w", dtype=tf.float32)
        b = tf.Variable(tf.constant([0.01]), name="b", dtype=tf.float32)
        b2 = tf.Variable(tf.constant([0.01]), name="b2", dtype=tf.float32)

      label_placeholders = self.add_label_placeholders(graph, name_scopes)
      weight_placeholders = self.add_example_weight_placeholders(
          graph, name_scopes)
      if training:
        graph.queue = tf.FIFOQueue(
            capacity=5,
            dtypes=[tf.float32] *
            (len(label_placeholders) + len(weight_placeholders) + 1))
        graph.enqueue = graph.queue.enqueue([mol_features] + label_placeholders
                                            + weight_placeholders)
        queue_outputs = graph.queue.dequeue()
        labels = queue_outputs[1:len(label_placeholders) + 1]
        weights = queue_outputs[len(label_placeholders) + 1:]
        features = queue_outputs[0]
      else:
        labels = label_placeholders
        weights = weight_placeholders
        features = mol_features

      for count in range(self.n_tasks):
        similarity = features[:, 2 * K * count:(2 * K * count + K)]
        ys = tf.to_int32(features[:, (2 * K * count + K):2 * K * (count + 1)])
        R = b + W[0] * similarity + W[1] * tf.constant(
            np.arange(K) + 1, dtype=tf.float32)
        R = tf.sigmoid(R)
        z = tf.reduce_sum(R * tf.gather(V, ys), axis=1) + b2
        output.append(tf.reshape(z, shape=[-1, 1]))
    return (output, labels, weights)
