"""TensorFlow implementation of fully connected networks. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf

from deepchem.utils.save import log
from deepchem.models.tensorflow_models import TensorflowGraph
from deepchem.models.tensorflow_models import TensorflowGraphModel
from deepchem.models.tensorflow_models.lr import TensorflowLogisticRegression

class TensorflowMultiTaskIRVClassifier(TensorflowLogisticRegression):

  def __init__(self, n_tasks, K=10, logdir=None, penalty=0.0, n_classes=2,
               penalty_type="l2", learning_rate=0.001, momentum=.8, 
               optimizer="adam", batch_size=50, verbose=True, seed=None,
               **kwargs):

    """Initialize TensorflowMultiTaskFitTransformRegressor
       
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    K: int
      Number of nearest neighbours used in classification
    logdir: str
      Location to save data
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
    fit_transformers: list
      List of dc.trans.FitTransformer objects

    """

    self.n_tasks = n_tasks
    self.K = K    
    self.n_features = 2*self.K*self.n_tasks
    print("n_features after fit_transform: %d" % int(self.n_features))
    TensorflowGraphModel.__init__(self, n_tasks, self.n_features, logdir=logdir, 
	       layer_sizes=None, weight_init_stddevs=None, bias_init_consts=None, 
              penalty=penalty, penalty_type=penalty_type, dropouts=None, 
	       n_classes=n_classes, learning_rate=learning_rate, 
             momentum=momentum, optimizer=optimizer, 
	       batch_size=batch_size, pad_batches=False, verbose=verbose, seed=seed, 
	       **kwargs)

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    K = self.K
    with graph.as_default():
      output = []
      with placeholder_scope:
        self.features = tf.placeholder(
            tf.float32, shape=[None, self.n_features], name='mol_features')
      with tf.name_scope('variable'):
        V = tf.Variable(tf.constant([0.01,1.]), name="vote", dtype=tf.float32)
        W = tf.Variable(tf.constant([1., 1.]), name="w", dtype=tf.float32)
        b = tf.Variable(tf.constant([0.01]), name="b", dtype=tf.float32)
        b2 = tf.Variable(tf.constant([0.01]), name="b2", dtype=tf.float32)
      for count in range(self.n_tasks):
        similarity = self.features[:, 2*K*count:(2*K*count+K)]
        ys = tf.to_int32(self.features[:, (2*K*count+K):2*K*(count+1)])
        R = b+W[0]*similarity+W[1]*tf.constant(np.arange(K)+1, dtype=tf.float32)
        R = tf.sigmoid(R)
        z = tf.reduce_sum(R * tf.gather(V,ys), axis=1) + b2
        output.append(tf.reshape(z, shape=[-1,1]))
    return output
  
