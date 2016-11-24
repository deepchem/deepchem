from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from deepchem.models.tensorflow_models import TensorflowGraph
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.models.tensorflow_models import model_ops

class ProgressiveMultitaskRegressor(TensorflowMultiTaskRegressor):
  """Implements a progressive multitask neural network.
  
  Progressive Networks: https://arxiv.org/pdf/1606.04671v3.pdf

  Progressive networks allow for multitask learning where each task
  gets a new column of weights. As a result, there is no exponential
  forgetting where previous tasks are ignored.
  """
  def __init__(self, n_tasks, n_features, alpha_init_stddevs=[.02], **kwargs):
    """Creates a progressive network.
  
    Only listing parameters specific to progressive networks here.

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_features: int
      Number of input features
    alpha_init_stddevs: list
      List of standard-deviations for alpha in adapter layers.
    """
    self.alpha_init_stddevs = alpha_init_stddevs
    super(ProgressiveMultitaskRegressor, self).__init__(
        n_tasks, n_features, **kwargs)

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    Parameters
    ----------
    graphs: tf.Graph
      Graph to build this model within.
    name_scopes: dict 
      Stores all the name scopes for this model.
    training: bool
      Indicates whether this graph is to be constructed in training
      or evaluation mode. Mainly used for dropout
    """
    # Create the scope for placeholders
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    # Create the namescopes for each task
    task_scopes = {}
    for task in range(self.n_tasks):
      task_root = "task%d" % task
      task_scopes[task] = TensorflowGraph.shared_name_scope(
          "task%d" % task, graph, name_scopes)

    with graph.as_default():
      with placeholder_scope:
        self.features = tf.placeholder(
            tf.float32, shape=[None, self.n_features], name='features')

    layer_sizes = self.layer_sizes
    alpha_init_stddevs = self.alpha_init_stddevs
    weight_init_stddevs = self.weight_init_stddevs
    bias_init_consts = self.bias_init_consts
    dropouts = self.dropouts

    # Consistency check
    lengths_set = {
        len(layer_sizes),
        len(weight_init_stddevs),
        len(alpha_init_stddevs),
        len(bias_init_consts),
        len(dropouts),
        }
    assert len(lengths_set) == 1, "All layer params must have same length."
    n_layers = lengths_set.pop()

    all_layers = {}
    for i in range(n_layers):
      for task in range(self.n_tasks):
        task_scope = task_scopes[task]
        print("Adding weights for task %d, layer %d" % (task, i))
        with task_scope:
          # Create the non-linear adapter
          if i == 0:
            prev_layer = self.features
          else:
            if task > 0:
              prev_layers = []
              # Iterate over all previous tasks.
              for prev_task in range(task-1):
                prev_layers.append(all_layers[(i-1, prev_task)])
              # prev_layers is a list with elements of size
              # (batch_size, layer_sizes[i-1])
              prev_layer = tf.concat(1, prev_layers)
              alpha = tf.Variable(tf.truncated_normal(
                  [1,], stddev=alpha_init_stddevs[i]))
              prev_layer = tf.mul(alpha, prev_layer)
              prev_layer_size = (task-1)*layer_sizes[i-1]
              V = tf.Variable(
                  tf.truncated_normal(
                      shape=[prev_layer_size, layer_sizes[i-1]],
                      stddev=weight_init_stddevs[i]),
                  name="V_layer_%d_task%d" % (i, task), dtype=tf.float32)
              b_lat = tf.Variable(
                  tf.constant(value=bias_init_consts[i],
                              shape=[prev_layer_size]),
                  name='b_lat_layer_%d_task%d' % (i, task),
                  dtype=tf.float32)
              prev_layer = tf.nn.xw_plus_b(prev_layer, V, b_lat)
              U = tf.Variable(
                  tf.truncated_normal(
                      shape=[layer_sizes[i-1], layer_sizes[i]],
                      stddev=weight_init_stddevs[i]),
                  name="U_layer_%d_task%d" % (i, task), dtype=tf.float32)
              lateral_contrib = tf.matmul(U, prev_layer)
      
          if i == 0:
            prev_layer_size = self.n_features
          else:
            prev_layer_size = layer_sizes[i-1]
          W = tf.Variable(
              tf.truncated_normal(
                  shape=[prev_layer_size, layer_sizes[i]],
                  stddev=weight_init_stddevs[i]),
              name='W_layer_%d_task%d' % (i, task), dtype=tf.float32)
          b = tf.Variable(tf.constant(value=bias_init_consts[i],
                          shape=[layer_sizes[i]]),
                          name='b_layer_%d_task%d' % (i, task), dtype=tf.float32)
          ##################################################### DEBUG
          print("prev_layer")
          print(prev_layer)
          print("prev_layer_size")
          print(prev_layer_size)
          print("W")
          print(W)
          print("b")
          print(b)
          ##################################################### DEBUG
          layer = tf.nn.xw_plus_b(prev_layer, W, b)
          if task > 0:
            layer = tf.add(layer, lateral_contrib)
          layer = tf.nn.relu(layer)
          # layer is of shape (batch_size, layer_sizes[i])
          layer = model_ops.dropout(layer, dropouts[i], training)
          all_layers[(i, task)] = layer
    # Gather up all the outputs to return.
    outputs = [all_layers[(i, task)] for task in range(self.n_tasks)]
    return outputs

  def add_training_costs(self, graph, name_scopes, outputs, labels, weights):
    """Adds the training costs for each task.
    
    Since each task is trained separately, each task is optimized w.r.t a separate
    task.

    TODO(rbharath): Figure out how to support weight decay for this model.
    Since each task is trained separately, weight decay should only be used
    on weights in column for that task.

    Parameters
    ----------
    graph: tf.Graph
      Graph for the model.
    name_scopes: dict
      Contains all the scopes for model
    outputs: list
      List of output tensors from model.
    weights: list
      List of weight placeholders for model.
    """
    with graph.as_default():
      epsilon = 1e-3  # small float to avoid dividing by zero
      weighted_costs = []  # weighted costs for each example
      gradient_costs = []  # costs used for gradient calculation

      with TensorflowGraph.shared_name_scope('costs', graph, name_scopes):
        for task in range(self.n_tasks):
          with TensorflowGraph.shared_name_scope(
              'cost_%d' % task, graph, name_scopes):
            with tf.name_scope('weighted'):
              weighted_cost = self.cost(output[task], labels[task],
                                        weights[task])
              weighted_costs.append(weighted_cost)

            with tf.name_scope('gradient'):
              # Note that we divide by the batch size and not the number of
              # non-zero weight examples in the batch.  Also, instead of using
              # tf.reduce_mean (which can put ops on the CPU) we explicitly
              # calculate with div/sum so it stays on the GPU.
              task_cost = tf.div(tf.reduce_sum(weighted_cost), self.batch_size)
              task_costs[task] = task_cost 

    return task_costs

  def add_output_ops(self, graph, outputs):
    """No-op for regression models since no softmax."""
    return outputs

  def fit(self, dataset, **kwargs):
    """Fit the model.

    Progressive networks are fit by training one task at a time. Iteratively
    fits one task at a time with other weights frozen.

    Parameters
    ---------- 
    dataset: dc.data.Dataset
      Dataset object holding training data 

    Raises
    ------
    AssertionError
      If model is not in training mode.
    """
    for task in range(self.n_tasks):
      self.fit_task(dataset, **kwargs)

  def get_training_op(self, graph, loss, task):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Returns:
    A training op.
    """
    with graph.as_default():
      task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    self.task_scopes[task])
      opt = model_ops.optimizer(self.optimizer, self.learning_rate, self.momentum)
      return opt.minimize(loss, name='train', var_list=task_vars)


  def fit_task(self, dataset, task, nb_epoch=10, pad_batches=False,
               max_checkpoints_to_keep=5, log_every_N_batches=50):
    """Fit the model.

    Fit one task.

    TODO(rbharath): Figure out if the logging will work correctly with the
    global_step set as it is.

    Parameters
    ---------- 
    dataset: dc.data.Dataset
      Dataset object holding training data 
    task: int
      The index of the task to train on.
    nb_epoch: 10
      Number of training epochs.
    pad_batches: bool
      Whether or not to pad each batch to exactly be of size batch_size.
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
    log("Training task %d for %d epochs" % (task, nb_epoch), self.verbosity)
    with self.train_graph.graph.as_default():
      task_train_op = self.get_training_op(
          self.train_graph.graph, self.train_graph.loss, task)
      with self._get_shared_session(train=True) as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=(task-1)*nb_epoch)
        for epoch in range(nb_epoch):
          avg_loss, n_batches = 0., 0
          for ind, (X_b, y_b, w_b, ids_b) in enumerate(
              # Turns out there are valid cases where we don't want pad-batches
              # on by default.
              #dataset.iterbatches(batch_size, pad_batches=True)):
              dataset.iterbatches(self.batch_size, pad_batches=pad_batches)):
            if ind % log_every_N_batches == 0:
              log("On batch %d" % ind, self.verbosity)
            # Run training op.
            feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
            fetches = self.train_graph.output + [
                task_train_op, self.train_graph.loss[task]]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            output = fetched_values[:len(self.train_graph.output)]
            loss = fetched_values[-1]
            avg_loss += loss
            y_pred = np.squeeze(np.array(output))
            y_b = y_b.flatten()
            n_batches += 1
          saver.save(sess, self._save_path, global_step=(task-1)*nb_epoch+epoch)
          avg_loss = float(avg_loss)/n_batches
          log('Ending epoch %d: Average loss %g' % (epoch, avg_loss), self.verbosity)
      ############################################################## TIMING
      time2 = time.time()
      print("TIMING: model fitting took %0.3f s" % (time2-time1),
            self.verbosity)
      ############################################################## TIMING
    # Always save a final checkpoint when complete.
    saver.save(sess, self._save_path, global_step=self.n_task*nb_epoch)
