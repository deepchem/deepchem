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
               fit_transformers=[], **kwargs):

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

    self.fit_transformers = fit_transformers
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
            tf.float32, shape=[None, self.n_features], name='features')
      with tf.name_scope('variable'):
        V = tf.Variable(tf.constant([0.01,1.]), name="vote", dtype=tf.float32)
        W = tf.Variable(tf.constant([1., 1.]), name="w", dtype=tf.float32)
        b = tf.Variable(tf.constant([0.01]), name="b", dtype=tf.float32)
        b2 = tf.Variable(tf.constant([0.01]), name="b2", dtype=tf.float32)
      for count in self.n_tasks:
        similarity = self.features[:, 2*K*count:(2*K*count+K)]
        ys = tf.to_int32(self.features[:, (2*K*count+K):2*K*(count+1)])
        R = b+W[0]*similarity+W[1]*tf.constant(np.arange(K)+1, dtype=tf.float32)
        R = tf.sigmoid(R)
        z = tf.reduce_sum(R * tf.gather(V,ys), axis=1) + b2
        output.append(z)
    return output
  
  def fit(self, dataset, nb_epoch=10, max_checkpoints_to_keep=5, log_every_N_batches=50, **kwargs):
    """Perform fit transformations on each minibatch. Fit the model.

    Parameters
    ---------- 
    dataset: dc.data.Dataset
      Dataset object holding training data 
    nb_epoch: 10
      Number of training epochs.
    max_checkpoints_to_keep: int
      Maximum number of checkpoints to keep; older checkpoints will be deleted.
    log_every_N_batches: int
      Report every N batches. Useful for training on very large datasets,
      where epochs can take long time to finish.

    Raises
    ------
    AssertionError
      If model is not in training mode.
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    log("Training for %d epochs" % nb_epoch, self.verbose)
    with self.train_graph.graph.as_default():
      train_op = self.get_training_op(
          self.train_graph.graph, self.train_graph.loss)
      with self._get_shared_session(train=True) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=0)
        for epoch in range(nb_epoch):
          avg_loss, n_batches = 0., 0
          for ind, (X_b, y_b, w_b, ids_b) in enumerate(
              dataset.iterbatches(self.batch_size, pad_batches=self.pad_batches)):
            if ind % log_every_N_batches == 0:
              log("On batch %d" % ind, self.verbose)
            for transformer in self.fit_transformers:
              X_b = transformer.X_transform(X_b)	
            # Run training op.
            feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
            fetches = self.train_graph.output + [
                train_op, self.train_graph.loss]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            output = fetched_values[:len(self.train_graph.output)]
            loss = fetched_values[-1]
            avg_loss += loss
            y_pred = np.squeeze(np.array(output))
            y_b = y_b.flatten()
            n_batches += 1
          saver.save(sess, self._save_path, global_step=epoch)
          avg_loss = float(avg_loss)/n_batches
          log('Ending epoch %d: Average loss %g' % (epoch, avg_loss), self.verbose)
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=epoch+1)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2-time1),
          self.verbose)
    ############################################################## TIMING

  def predict_on_batch(self, X):
    """Return model output for the provided input. Each example is evaluated
        self.n_evals times.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: dc.data.Dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
        labels: True labels.
        weights: Example weights.
      Note that the output and labels arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    X_evals = []
    for i in range(self.n_evals):
      X_t = X
      for transformer in self.fit_transformers:
        X_t = transformer.X_transform(X_t)
      X_evals.append(X_t)
    len_unpadded = len(X_t)
    if self.pad_batches:
      for i in range(self.n_evals):
        X_evals[i] = pad_features(self.batch_size, X_evals[i])
    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():

      # run eval data through the model
      n_tasks = self.n_tasks
      outputs = []
      with self._get_shared_session(train=False).as_default():

        n_samples = len(X_evals[0])
        for i in range(self.n_evals):

          output = []
          feed_dict = self.construct_feed_dict(X_evals[i])
          data = self._get_shared_session(train=False).run(
              self.eval_graph.output, feed_dict=feed_dict)
          batch_outputs = np.asarray(data[:n_tasks], dtype=float)
          # reshape to batch_size x n_tasks x ...
          if batch_outputs.ndim == 3:
            batch_outputs = batch_outputs.transpose((1, 0, 2))
          elif batch_outputs.ndim == 2:
            batch_outputs = batch_outputs.transpose((1, 0))
          # Handle edge case when batch-size is 1.
          elif batch_outputs.ndim == 1:
            n_samples = len(X)
            batch_outputs = batch_outputs.reshape((n_samples, n_tasks))
          else:
            raise ValueError(
                'Unrecognized rank combination for output: %s' %
                (batch_outputs.shape))
          # Prune away any padding that was added
          batch_outputs = batch_outputs[:n_samples]
          output.append(batch_outputs)

          outputs.append(np.squeeze(np.concatenate(output)))
	  
    outputs = np.mean(np.array(outputs), axis=0)
    outputs = np.copy(outputs)

    # Handle case of 0-dimensional scalar output
    if len(outputs.shape) > 0:
      return outputs[:len_unpadded]
    else:
      outputs = np.reshape(outputs, (1,))
      return outputs