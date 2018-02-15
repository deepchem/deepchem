from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf
import collections

from deepchem.utils.save import log
from deepchem.metrics import to_one_hot
from deepchem.metrics import from_one_hot
from deepchem.models.tensorgraph.tensor_graph import TensorGraph, TFWrapper
from deepchem.models.tensorgraph.layers import Layer, Feature, Label, Weights, \
    WeightedError, Dense, Dropout, WeightDecay, Reshape, SoftMaxCrossEntropy, \
    L2Loss, ReduceSum, Concat, Stack


class Activate(Layer):
  """ Compute the activation of input: f(x) = activate_fn(x)
  Only one input is allowed, output will have the same shape as input
  """

  def __init__(self, activation_fn, in_layers=None, **kwargs):
    self.activation_fn = activation_fn
    super(Activate, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Activate must have a single input layer.")
    parent = inputs[0]
    out_tensor = self.activation_fn(parent)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
  
class ProgressiveMultitaskRegressor(TensorGraph):
  """Implements a progressive multitask neural network.
  
  Progressive Networks: https://arxiv.org/pdf/1606.04671v3.pdf

  Progressive networks allow for multitask learning where each task
  gets a new column of weights. As a result, there is no exponential
  forgetting where previous tasks are ignored.

  """

  def __init__(self,
               n_tasks, 
               n_features, 
               alpha_init_stddevs=[.02],
               layer_sizes=[1000],
               weight_init_stddevs=0.02,
               bias_init_consts=1.0,
               weight_decay_penalty=0.0,
               weight_decay_penalty_type="l2",
               dropouts=0.5,
               activation_fns=tf.nn.relu,
               **kwargs):
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
    layer_sizes: list
      the size of each dense layer in the network.  The length of this list determines the number of layers.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization of each layer.  The length
      of this list should equal len(layer_sizes)+1.  The final element corresponds to the output layer.
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    bias_init_consts: list or float
      the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes)+1.
      The final element corresponds to the output layer.  Alternatively this may be a single value instead of a list,
      in which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float
      the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    activation_fns: list or object
      the Tensorflow activation function to apply to each layer.  The length of this list should equal
      len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.
    """
    
    super(ProgressiveMultitaskRegressor, self).__init__(**kwargs)
    self.alpha_init_stddevs = alpha_init_stddevs
    self.n_tasks = n_tasks
    self.n_features = n_features
    self.layer_sizes = layer_sizes
    n_layers = len(layer_sizes)
    if not isinstance(weight_init_stddevs, collections.Sequence):
      weight_init_stddevs = [weight_init_stddevs] * n_layers
    if not isinstance(bias_init_consts, collections.Sequence):
      bias_init_consts = [bias_init_consts] * n_layers
    if not isinstance(dropouts, collections.Sequence):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, collections.Sequence):
      activation_fns = [activation_fns] * n_layers

    # Add the input features.
    mol_features = Feature(shape=(None, n_features))
    prev_layer = mol_features

    all_layers = {}
    for i in range(n_layers):
      for task in range(self.n_tasks):
        if i == 0:
          prev_layer = self.mol_features
        else:
          prev_layer = all_layers[(i - 1, task)]
          if task > 0:
            lateral_contrib = self.add_adapter(all_layers, task, i)
        layer = Dense(
            in_layers=[prev_layer],
            out_channels=layer_sizes[i],
            activation_fn=None,
            weights_initializer=TFWrapper(
                tf.truncated_normal_initializer, stddev=weight_init_stddevs[i]),
            biases_initializer=TFWrapper(
                tf.constant_initializer, value=bias_init_consts[i]))
        if i > 0 and task > 0:
          layer = layer + lateral_contrib
        layer = Activate(activation_fns[i], in_layers=[layer])
        if dropouts[i] > 0.0:
          layer = Dropout(dropouts[0], in_layers=[layer])
        all_layers[(i, task)] = layer

    outputs = []
    for task in range(self.n_tasks):
      prev_layer = all_layers[(n_layers - 1, task)]
      #lateral_contrib = tf.squeeze(
      #    self.add_adapter(all_layers, task, i + 1))
      layer = Dense(
          in_layers=[prev_layer], 
          out_channels=1,
          weights_initializer=TFWrapper(
              tf.truncated_normal_initializer,
              stddev=weight_init_stddevs[-1]),
          biases_initializer=TFWrapper(
              tf.constant_initializer, value=bias_init_consts[-1]))
      if task > 0:
          layer = layer + lateral_contrib
      outputs.append(layer)
    output = Concat(in_layers=outputs)
    self.add_output(output)
    labels = Label(shape=(None, n_tasks))
    weights = Weights(shape=(None, n_tasks))
    weighted_loss = ReduceSum(L2Loss(in_layers=[labels, output, weights]))
    if weight_decay_penalty != 0.0:
      weighted_loss = WeightDecay(
          weight_decay_penalty,
          weight_decay_penalty_type,
          in_layers=[weighted_loss])
    self.set_loss(weighted_loss)

      



  def add_adapter(self, all_layers, task, layer_num):
    """Add an adapter connection for given task/layer combo"""
    i = layer_num
    prev_layers = []
    # Handle output layer
    if i < len(self.layer_sizes):
      layer_sizes = self.layer_sizes
      alpha_init_stddev = self.alpha_init_stddevs[i]
      weight_init_stddev = self.weight_init_stddevs[i]
      bias_init_const = self.bias_init_consts[i]
    elif i == len(self.layer_sizes):
      layer_sizes = self.layer_sizes + [1]
      alpha_init_stddev = self.alpha_init_stddevs[-1]
      weight_init_stddev = self.weight_init_stddevs[-1]
      bias_init_const = self.bias_init_consts[-1]
    else:
      raise ValueError("layer_num too large for add_adapter.")
    # Iterate over all previous tasks.
    for prev_task in range(task):
      prev_layers.append(all_layers[(i - 1, prev_task)])
    # prev_layers is a list with elements of size
    # (batch_size, layer_sizes[i-1])
    prev_layer = tf.concat(axis=1, values=prev_layers)
    alpha = tf.Variable(
        tf.truncated_normal([
            1,
        ], stddev=alpha_init_stddev),
        name="alpha_layer_%d_task%d" % (i, task))
    prev_layer = tf.multiply(alpha, prev_layer)
    prev_layer_size = task * layer_sizes[i - 1]
    print("Creating V_layer_%d_task%d of shape %s" %
          (i, task, str([prev_layer_size, layer_sizes[i - 1]])))
    V = tf.Variable(
        tf.truncated_normal(
            shape=[prev_layer_size, layer_sizes[i - 1]],
            stddev=weight_init_stddev),
        name="V_layer_%d_task%d" % (i, task),
        dtype=tf.float32)
    print("Creating b_lat_layer_%d_task%d of shape %s" %
          (i, task, str([layer_sizes[i - 1]])))
    b_lat = tf.Variable(
        tf.constant(value=bias_init_const, shape=[layer_sizes[i - 1]]),
        name='b_lat_layer_%d_task%d' % (i, task),
        dtype=tf.float32)
    prev_layer = tf.matmul(prev_layer, V) + b_lat
    print("Creating U_layer_%d_task%d of shape %s" %
          (i, task, str([layer_sizes[i - 1], layer_sizes[i]])))
    U = tf.Variable(
        tf.truncated_normal(
            shape=[layer_sizes[i - 1], layer_sizes[i]],
            stddev=weight_init_stddev),
        name="U_layer_%d_task%d" % (i, task),
        dtype=tf.float32)
    return tf.matmul(prev_layer, U)

  def get_training_op(self, graph, loss):
    """Get training op for applying gradients to variables.

    Subclasses that need to do anything fancy with gradients should override
    this method.

    Returns:
    A training op.
    """
    with graph.as_default():
      opt = model_ops.optimizer(self.optimizer, self.learning_rate,
                                self.momentum)
      return opt.minimize(loss, name='train')

  def add_training_costs(self, graph, name_scopes, output, labels, weights):
    with graph.as_default():
      epsilon = 1e-3  # small float to avoid dividing by zero
      weighted_costs = []  # weighted costs for each example
      gradient_costs = []  # costs used for gradient calculation

      with TensorflowGraph.shared_name_scope('costs', graph, name_scopes):
        for task in range(self.n_tasks):
          task_str = str(task).zfill(len(str(self.n_tasks)))
          with TensorflowGraph.shared_name_scope('cost_{}'.format(task_str),
                                                 graph, name_scopes):
            with tf.name_scope('weighted'):
              weighted_cost = self.cost(output[task], labels[task],
                                        weights[task])
              weighted_costs.append(weighted_cost)

            with tf.name_scope('gradient'):
              # Note that we divide by the batch size and not the number of
              # non-zero weight examples in the batch.  Also, instead of using
              # tf.reduce_mean (which can put ops on the CPU) we explicitly
              # calculate with div/sum so it stays on the GPU.
              gradient_cost = tf.div(
                  tf.reduce_sum(weighted_cost), self.batch_size)
              gradient_costs.append(gradient_cost)

        # aggregated costs
        with TensorflowGraph.shared_name_scope('aggregated', graph,
                                               name_scopes):
          with tf.name_scope('gradient'):
            loss = tf.add_n(gradient_costs)

          # weight decay
          if self.penalty != 0.0:
            penalty = model_ops.weight_decay(self.penalty_type, self.penalty)
            loss += penalty

      return loss

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, n_features)
      y_b: np.ndarray of shape (batch_size, n_tasks)
      w_b: np.ndarray of shape (batch_size, n_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in range(self.n_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = y_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(np.zeros((self.batch_size,)))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones((self.batch_size,))
    return TensorflowGraph.get_feed_dict(orig_dict)

  def predict_on_batch(self, X, pad_batch=False):
    """Return model output for the provided input.

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
    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():

      # run eval data through the model
      n_tasks = self.n_tasks
      with self._get_shared_session(train=False).as_default():
        n_samples = len(X)
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session(train=False).run(
            self.eval_graph.output, feed_dict=feed_dict)
        # Shape (n_tasks, n__samples)
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
          raise ValueError('Unrecognized rank combination for output: %s' %
                           (batch_outputs.shape))
        outputs = np.squeeze(batch_outputs)

    return outputs

  def fit(self,
          dataset,
          tasks=None,
          close_session=True,
          max_checkpoints_to_keep=5,
          **kwargs):
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
    if tasks is None:
      tasks = range(self.n_tasks)
    with self.train_graph.graph.as_default():
      task_train_ops = {}
      for task in tasks:
        task_train_ops[task] = self.get_task_training_op(
            self.train_graph.graph, self.train_graph.loss, task)

      sess = self._get_shared_session(train=True)
      #with self._get_shared_session(train=True) as sess:
      sess.run(tf.global_variables_initializer())
      # Save an initial checkpoint.
      saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      saver.save(sess, self._save_path, global_step=0)
      for task in tasks:
        print("Fitting on task %d" % task)
        self.fit_task(sess, dataset, task, task_train_ops[task], **kwargs)
        saver.save(sess, self._save_path, global_step=task)
      # Always save a final checkpoint when complete.
      saver.save(sess, self._save_path, global_step=self.n_tasks)
      if close_session:
        sess.close()

  def get_task_training_op(self, graph, losses, task):
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
      task_root = "task%d_ops" % task
      task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, task_root)
      opt = model_ops.optimizer(self.optimizer, self.learning_rate,
                                self.momentum)
      return opt.minimize(task_loss, name='train', var_list=task_vars)

  def add_task_training_costs(self, graph, name_scopes, outputs, labels,
                              weights):
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
    task_costs = {}
    with TensorflowGraph.shared_name_scope('costs', graph, name_scopes):
      for task in range(self.n_tasks):
        with TensorflowGraph.shared_name_scope('cost_%d' % task, graph,
                                               name_scopes):
          weighted_cost = self.cost(outputs[task], labels[task], weights[task])

          # Note that we divide by the batch size and not the number of
          # non-zero weight examples in the batch.  Also, instead of using
          # tf.reduce_mean (which can put ops on the CPU) we explicitly
          # calculate with div/sum so it stays on the GPU.
          task_cost = tf.div(tf.reduce_sum(weighted_cost), self.batch_size)
          task_costs[task] = task_cost

    return task_costs

  def construct_task_feed_dict(self,
                               this_task,
                               X_b,
                               y_b=None,
                               w_b=None,
                               ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, n_features)
      y_b: np.ndarray of shape (batch_size, n_tasks)
      w_b: np.ndarray of shape (batch_size, n_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    n_samples = len(X_b)
    for task in range(self.n_tasks):
      if (this_task == task) and y_b is not None:
        #orig_dict["labels_%d" % task] = np.reshape(y_b[:, task], (n_samples, 1))
        orig_dict["labels_%d" % task] = np.reshape(y_b[:, task], (n_samples,))
      else:
        # Dummy placeholders
        #orig_dict["labels_%d" % task] = np.zeros((n_samples, 1))
        orig_dict["labels_%d" % task] = np.zeros((n_samples,))
      if (this_task == task) and w_b is not None:
        #orig_dict["weights_%d" % task] = np.reshape(w_b[:, task], (n_samples, 1))
        orig_dict["weights_%d" % task] = np.reshape(w_b[:, task], (n_samples,))
      else:
        # Dummy placeholders
        #orig_dict["weights_%d" % task] = np.zeros((n_samples, 1))
        orig_dict["weights_%d" % task] = np.zeros((n_samples,))
    return TensorflowGraph.get_feed_dict(orig_dict)

  def _get_shared_session(self, train):
    # allow_soft_placement=True allows ops without a GPU implementation
    # to run on the CPU instead.
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

  def fit_task(self,
               sess,
               dataset,
               task,
               task_train_op,
               nb_epoch=10,
               log_every_N_batches=50,
               checkpoint_interval=10):
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
    max_checkpoints_to_keep: int
      Maximum number of checkpoints to keep; older checkpoints will be deleted.
    log_every_N_batches: int
      Report every N batches. Useful for training on very large datasets,
      where epochs can take long time to finish.
    checkpoint_interval: int
      Frequency at which to write checkpoints, measured in epochs

    Raises
    ------
    AssertionError
      If model is not in training mode.
    """

    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    log("Training task %d for %d epochs" % (task, nb_epoch), self.verbose)
    for epoch in range(nb_epoch):
      avg_loss, n_batches = 0., 0
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          # Turns out there are valid cases where we don't want pad-batches
          # on by default.
          #dataset.iterbatches(batch_size, pad_batches=True)):
          dataset.iterbatches(self.batch_size, pad_batches=self.pad_batches)):
        if ind % log_every_N_batches == 0:
          log("On batch %d" % ind, self.verbose)
        feed_dict = self.construct_task_feed_dict(task, X_b, y_b, w_b, ids_b)
        fetches = self.train_graph.output + [
            task_train_op, self.train_graph.loss[task]
        ]
        fetched_values = sess.run(fetches, feed_dict=feed_dict)
        output = fetched_values[:len(self.train_graph.output)]
        loss = fetched_values[-1]
        avg_loss += loss
        y_pred = np.squeeze(np.array(output))
        y_b = y_b.flatten()
        n_batches += 1
      #if epoch%checkpoint_interval == checkpoint_interval-1:
      #  saver.save(sess, self._save_path, global_step=epoch)
      avg_loss = float(avg_loss) / n_batches
      log('Ending epoch %d: Average loss %g' % (epoch, avg_loss), self.verbose)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING
