from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf

from deepchem.utils.save import log
from deepchem.metrics import to_one_hot
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

  TODO(rbharath): This class is unnecessarily complicated. Can we simplify the
  structure of the code here?
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

  def construct_graph(self, training, seed):
    """Returns a TensorflowGraph object."""
    graph = tf.Graph() 

    # Lazily created by _get_shared_session().
    shared_session = None

    # Cache of TensorFlow scopes, to prevent '_1' appended scope names
    # when subclass-overridden methods use the same scopes.
    name_scopes = {}

    # Setup graph
    with graph.as_default():
      if seed is not None:
        tf.set_random_seed(seed)
      output = self.build(graph, name_scopes, training)
      labels = self.add_label_placeholders(graph, name_scopes)
      weights = self.add_example_weight_placeholders(graph, name_scopes)

    if training:
      loss = self.add_training_costs(graph, name_scopes, output, labels, weights)
    else:
      loss = None
      output = self.add_output_ops(graph, output)  # add softmax heads
    return TensorflowGraph(graph=graph,
                           session=shared_session,
                           name_scopes=name_scopes,
                           output=output,
                           labels=labels,
                           weights=weights,
                           loss=loss)

  def add_label_placeholders(self, graph, name_scopes):
    """Add Placeholders for labels for each task.

    This method creates the following Placeholders for each task:
      labels_%d: Label tensor with shape batch_size.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph, name_scopes)
    with graph.as_default():
      batch_size = self.batch_size
      labels = []
      with placeholder_scope:
        for task in range(self.n_tasks):
          labels.append(tf.identity(
              tf.placeholder(tf.float32, shape=[None, 1],
                             name='labels_%d' % task)))
    return labels

  def add_example_weight_placeholders(self, graph, name_scopes):
    """Add Placeholders for example weights for each task.

    This method creates the following Placeholders for each task:
      weights_%d: Label tensor with shape batch_size.

    Placeholders are wrapped in identity ops to avoid the error caused by
    feeding and fetching the same tensor.
    """
    weights = []
    placeholder_scope = TensorflowGraph.get_placeholder_scope(graph, name_scopes)
    with placeholder_scope:
      for task in range(self.n_tasks):
        weights.append(tf.identity(
            tf.placeholder(tf.float32, shape=[None, 1],
                           name='weights_%d' % task)))
    return weights

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
    self.task_scopes = {}
    for task in range(self.n_tasks):
      task_root = "task%d" % task
      self.task_scopes[task] = TensorflowGraph.shared_name_scope(
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
        task_scope = self.task_scopes[task]
        print("Adding weights for task %d, layer %d" % (task, i))
        with task_scope:
          # Create the non-linear adapter
          if i == 0:
            prev_layer = self.features
          else:
            pass
            ############################################################### DEBUG
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
              print("Creating V_layer_%d_task%d of shape %s" %
                    (i, task, str([prev_layer_size, layer_sizes[i-1]])))
              V = tf.Variable(
                  tf.truncated_normal(
                      shape=[prev_layer_size, layer_sizes[i-1]],
                      stddev=weight_init_stddevs[i]),
                  name="V_layer_%d_task%d" % (i, task), dtype=tf.float32)
              print("Creating b_lat_layer_%d_task%d of shape %s" %
                    (i, task, str([prev_layer_size])))
              b_lat = tf.Variable(
                  tf.constant(value=bias_init_consts[i],
                              shape=[prev_layer_size]),
                  name='b_lat_layer_%d_task%d' % (i, task),
                  dtype=tf.float32)
              prev_layer = tf.nn.xw_plus_b(prev_layer, V, b_lat)
              print("Creating V_layer_%d_task%d of shape %s" %
                    (i, task, str([layer_sizes[i-1], layer_sizes[i]])))
              U = tf.Variable(
                  tf.truncated_normal(
                      shape=[layer_sizes[i-1], layer_sizes[i]],
                      stddev=weight_init_stddevs[i]),
                  name="U_layer_%d_task%d" % (i, task), dtype=tf.float32)
              lateral_contrib = tf.matmul(U, prev_layer)
            ############################################################### DEBUG
      
          if i == 0:
            prev_layer_size = self.n_features
          else:
            prev_layer_size = layer_sizes[i-1]
          print("Creating W_layer_%d_task%d of shape %s" %
                (i, task, str([prev_layer_size, layer_sizes[i]])))
          W = tf.Variable(
              tf.truncated_normal(
                  shape=[prev_layer_size, layer_sizes[i]],
                  stddev=weight_init_stddevs[i]),
              name='W_layer_%d_task%d' % (i, task), dtype=tf.float32)
          print("Creating b_layer_%d_task%d of shape %s" %
                (i, task, str([layer_sizes[i]])))
          b = tf.Variable(tf.constant(value=bias_init_consts[i],
                          shape=[layer_sizes[i]]),
                          name='b_layer_%d_task%d' % (i, task), dtype=tf.float32)
          layer = tf.nn.xw_plus_b(prev_layer, W, b)
          #################################################### DEBUG
          print("layer")
          print(layer)
          #################################################### DEBUG
          if i > 0 and task > 0:
            layer = tf.add(layer, lateral_contrib)
          layer = tf.nn.relu(layer)
          #################################################### DEBUG
          print("layer")
          print(layer)
          #################################################### DEBUG
          # layer is of shape (batch_size, layer_sizes[i])
          layer = model_ops.dropout(layer, dropouts[i], training)
          #################################################### DEBUG
          print("layer")
          print(layer)
          #################################################### DEBUG
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
    ###################################################### DEBUG
    print("add_training_cost()")
    ###################################################### DEBUG
    task_costs = {}
    with graph.as_default():
      epsilon = 1e-3  # small float to avoid dividing by zero
      weighted_costs = []  # weighted costs for each example
      gradient_costs = []  # costs used for gradient calculation

      with TensorflowGraph.shared_name_scope('costs', graph, name_scopes):
        for task in range(self.n_tasks):
          with TensorflowGraph.shared_name_scope(
              'cost_%d' % task, graph, name_scopes):
            with tf.name_scope('weighted'):
              weighted_cost = self.cost(outputs[task], labels[task],
                                        weights[task])
              ###################################################### DEBUG
              print("weighted_cost")
              print(weighted_cost)
              print("outputs[task], labels[task], weights[task]")
              print(outputs[task], labels[task], weights[task])
              ###################################################### DEBUG
              weighted_costs.append(weighted_cost)

            with tf.name_scope('gradient'):
              # Note that we divide by the batch size and not the number of
              # non-zero weight examples in the batch.  Also, instead of using
              # tf.reduce_mean (which can put ops on the CPU) we explicitly
              # calculate with div/sum so it stays on the GPU.
              task_cost = tf.div(tf.reduce_sum(weighted_cost), self.batch_size)
              task_costs[task] = task_cost 
    ###################################################### DEBUG
    print("task_costs")
    print(task_costs)
    ###################################################### DEBUG

    return task_costs

  def add_output_ops(self, graph, outputs):
    """No-op for regression models since no softmax."""
    return outputs

  def fit(self, dataset, max_checkpoints_to_keep=5, **kwargs):
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
    with self.train_graph.graph.as_default():
      task_train_ops = {}
      for task in range(self.n_tasks):
        task_train_ops[task] = self.get_training_op(
            self.train_graph.graph, self.train_graph.loss, task)
      with self._get_shared_session(train=True) as sess:
        sess.run(tf.initialize_all_variables())
        # Save an initial checkpoint.
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        saver.save(sess, self._save_path, global_step=0)
        for task in range(self.n_tasks):
          ################################################## DEBUG
          print("FITTING ON TASK %d" % task)
          ################################################## DEBUG
          self.fit_task(sess, dataset, task, task_train_ops[task], **kwargs)
          saver.save(sess, self._save_path, global_step=task)
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=self.n_tasks)

  def get_training_op(self, graph, losses, task):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Parameters
    ----------
    graph: tf.Graph
      Graph for this op
    losses: dict
      Dictionary mapping task to losses

    Returns
    -------
    A training op.
    """
    with graph.as_default():
      task_loss = losses[task]
      task_root = "task%d" % task
      task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, task_root)
      opt = model_ops.optimizer(self.optimizer, self.learning_rate, self.momentum)
      ######################################################## DEBUG
      print("get_training_op()")
      print("task")
      print(task)
      print("task_root")
      print(task_root)
      print("[task_var.name for task_var in task_vars]")
      print([task_var.name for task_var in task_vars])
      ######################################################## DEBUG
      return opt.minimize(task_loss, name='train', var_list=task_vars)

  def _get_shared_session(self, train):
    # allow_soft_placement=True allows ops without a GPU implementation
    # to run on the CPU instead.
    ############################################### DEBUG
    print("_get_shared_session")
    print("self.train_graph.session")
    print(self.train_graph.session)
    ############################################### DEBUG
    if train:
      if not self.train_graph.session:
        config = tf.ConfigProto(allow_soft_placement=True)
        self.train_graph.session = tf.Session(config=config)
      return self.train_graph.session
    else:
      if not self.eval_graph.session:
        config = tf.ConfigProto(allow_soft_placement=True)
        self.eval_graph.session = tf.Session(config=config)
      return self.eval_graph.session


  def fit_task(self, sess, dataset, task, task_train_op, nb_epoch=10, pad_batches=False,
               log_every_N_batches=50):
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
    ##with self.train_graph.graph.as_default():
    #  #task_train_op = self.get_training_op(
    #  #    self.train_graph.graph, self.train_graph.loss, task)
    #  #with self._get_shared_session(train=True) as sess:
    #    #sess.run(tf.initialize_all_variables())
    #    #saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
    #    ## Save an initial checkpoint.
    #    #saver.save(sess, self._save_path, global_step=0)
    for epoch in range(nb_epoch):
      avg_loss, n_batches = 0., 0
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          # Turns out there are valid cases where we don't want pad-batches
          # on by default.
          #dataset.iterbatches(batch_size, pad_batches=True)):
          dataset.iterbatches(self.batch_size, pad_batches=pad_batches)):
        if ind % log_every_N_batches == 0:
          log("On batch %d" % ind, self.verbosity)
        feed_dict = self.construct_feed_dict(task, X_b, y_b, w_b, ids_b)
        fetches = self.train_graph.output + [
            task_train_op, self.train_graph.loss[task]]
        fetched_values = sess.run(fetches, feed_dict=feed_dict)
        output = fetched_values[:len(self.train_graph.output)]
        loss = fetched_values[-1]
        avg_loss += loss
        y_pred = np.squeeze(np.array(output))
        y_b = y_b.flatten()
        n_batches += 1
      #saver.save(sess, self._save_path, global_step=epoch)
      avg_loss = float(avg_loss)/n_batches
      log('Ending epoch %d: Average loss %g' % (epoch, avg_loss), self.verbosity)
    #    ## Always save a final checkpoint when complete.
    #    #saver.save(sess, self._save_path, global_step=nb_epoch)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2-time1),
          self.verbosity)
    ############################################################## TIMING

  def construct_feed_dict(self, task, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, n_features)
      y_b: np.ndarray of shape (batch_size, n_tasks)
      w_b: np.ndarray of shape (batch_size, n_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """ 
    orig_dict = {}
    orig_dict["features"] = X_b
    n_samples = len(X_b)
    if y_b is not None:
      orig_dict["labels_%d" % task] = np.reshape(y_b[:, task], (n_samples, 1))
    else:
      # Dummy placeholders
      orig_dict["labels_%d" % task] = np.squeeze(
          np.zeros((self.batch_size,)))
    if w_b is not None:
      orig_dict["weights_%d" % task] = np.reshape(w_b[:, task], (n_samples, 1))
    else:
      # Dummy placeholders
      orig_dict["weights_%d" % task] = np.ones(
          (self.batch_size,)) 
    ################################################################ DEBUG
    #for key, value in orig_dict.iteritems():
    #  print("key")
    #  print(key)
    #  print("value.shape")
    #  print(value.shape)
    ################################################################ DEBUG
    return TensorflowGraph.get_feed_dict(orig_dict)
