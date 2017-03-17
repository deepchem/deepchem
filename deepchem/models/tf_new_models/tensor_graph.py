import networkx as nx
import tensorflow as tf
import time

from deepchem.models.models import Model
from nn.copy import Layer


class TensorGraph(Model):

  def __init__(self, data_dir, **kwargs):
    self.data_dir = data_dir
    self.nxgraph = nx.DiGraph()
    self.features = None
    self.labels = None
    self.outputs = None
    self.train_op = None
    self.loss = None
    self.graph = tf.Graph()
    super().__init__(**kwargs)

  def add_layer(self, layer, parents=list()):
    with self.graph.as_default():
      # TODO(LESWING) Assert layer.name not already in nxgraph to not allow duplicates
      self.nxgraph.add_node(layer.name)
      for parent in parents:
        self.nxgraph.add_edge(parent.name, layer.name)
      # TODO(LESWING) do this lazily and call in Topological sort order after call to "build"
      layer.set_parents(parents)
      return layer

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          learning_rate=.001,
          batch_size=50,
          checkpoint_interval=10):
    """Trains the model for a fixed number of epochs.

    TODO(rbharath0: This is mostly copied from TensorflowGraphModel. Should
    eventually refactor both together.

    Parameters
    ----------
    dataset: dc.data.Dataset
    nb_epoch: 10
      Number of training epochs.
      Dataset object holding training data
        batch_size: integer. Number of samples per gradient update.
        nb_epoch: integer, the number of epochs to train the model.
        verbose: 0 for no logging to stdout,
            1 for progress bar logging, 2 for one log line per epoch.
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)
    checkpoint_interval: int
      Frequency at which to write checkpoints, measured in epochs
    """
    with self.graph.as_default():
      time1 = time.time()
      print("Training for %d epochs" % nb_epoch)
      self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
      saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(self.data_dir))
        #sess.run(tf.global_variables_initializer())
        # Save an initial checkpoint.
        #saver.save(sess, self.data_dir, global_step=0)
        for epoch in range(nb_epoch):
          avg_loss, n_batches = 0., 0
          # TODO(rbharath): Don't support example weighting yet.
          for ind, (X_b, y_b, w_b, ids_b) in enumerate(
              dataset.iterbatches(batch_size, pad_batches=True)):
            if ind % log_every_N_batches == 0:
              print("On batch %d" % ind)
            feed_dict = {self.features: X_b, self.labels: y_b}
            fetches = [self.outputs] + [self.train_op, self.loss]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            loss = fetched_values[-1]
            avg_loss += loss
            n_batches += 1
          if epoch % checkpoint_interval == checkpoint_interval - 1:
            pass
            #saver.save(sess, self.data_dir, global_step=epoch)
          avg_loss = float(avg_loss) / n_batches
          print('Ending epoch %d: Average loss %g' % (epoch, avg_loss))
          # Always save a final checkpoint when complete.
          #saver.save(sess, self.data_dir, global_step=epoch + 1)
      ############################################################## TIMING
      time2 = time.time()
      print("TIMING: model fitting took %0.3f s" % (time2 - time1))
      ############################################################## TIMING


class Conv1DLayer(Layer):

  def __init__(self, width, out_channels, **kwargs):
    self.width = width
    self.out_channels = out_channels
    self.out_tensor = None
    super().__init__(**kwargs)

  def set_parents(self, parents):
    if len(parents) != 1:
      raise ValueError("Only One Parent to conv1D over")
    parent = parents[0]
    if len(parent.out_tensor.get_shape()) != 3:
      raise ValueError("Parent tensor must be (batch, width, channel)")
    parent_shape = parent.out_tensor.get_shape()
    parent_channel_size = parent_shape[2].value
    with tf.name_scope(self.name):
      f = tf.Variable(
          tf.random_normal([self.width, parent_channel_size, self.out_channels
                           ]))
      b = tf.Variable(tf.random_normal([self.out_channels]))
      t = tf.nn.conv1d(parent.out_tensor, f, stride=1, padding="SAME")
      t = tf.nn.bias_add(t, b)
      self.out_tensor = tf.nn.relu(t)


class Dense(Layer):

  def __init__(self, out_channels, **kwargs):
    self.out_channels = out_channels
    self.out_tensor = None
    super().__init__(**kwargs)

  def set_parents(self, parents):
    if len(parents) != 1:
      raise ValueError("Only One Parent to Dense over")
    parent = parents[0]
    if len(parent.out_tensor.get_shape()) != 2:
      raise ValueError("Parent tensor must be (batch, width)")
    parent_shape = parent.out_tensor.get_shape()
    with tf.name_scope(self.name):
      w = tf.random_normal(shape=(parent_shape[1].value, self.out_channels))
      b = tf.random_normal([self.out_channels])
      self.out_tensor = tf.matmul(parent.out_tensor, w) + b
    return self.out_tensor


class Flatten(Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def set_parents(self, parents):
    if len(parents) != 1:
      raise ValueError("Only One Parent to conv1D over")
    parent = parents[0]
    if len(parent.out_tensor.get_shape()) != 3:
      raise ValueError("Parent tensor must be (batch, width, channel)")
    parent_shape = parent.out_tensor.get_shape()
    vector_size = parent_shape[1].value * parent_shape[2].value
    parent_tensor = parent.out_tensor
    with tf.name_scope(self.name):
      self.out_tensor = tf.reshape(parent_tensor, shape=(-1, vector_size))


class CombineMeanStd(Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def set_parents(self, parents):
    if len(parents) != 2:
      raise ValueError("Must have two parents")
    mean_parent, std_parent = parents[0], parents[1]
    mean_parent_tensor, std_parent_tensor = mean_parent.out_tensor, std_parent.out_tensor
    with tf.name_scope(self.name):
      sample_noise = tf.random_normal(
          mean_parent_tensor.get_shape(), 0, 1, dtype=tf.float32)
      self.out_tensor = mean_parent_tensor + (std_parent_tensor * sample_noise)


class Repeat(Layer):

  def __init__(self, n_times, **kwargs):
    self.n_times = n_times
    super().__init__(**kwargs)

  def set_parents(self, parents):
    if len(parents) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = parents[0].out_tensor
    with tf.name_scope(self.name):
      t = tf.expand_dims(parent_tensor, 1)
      pattern = tf.pack([1, self.n_times, 1])
      self.out_tensor = tf.tile(t, pattern)


class GRU(Layer):

  def __init__(self, n_hidden, out_channels, batch_size, **kwargs):
    self.n_hidden = n_hidden
    self.out_channels = out_channels
    self.batch_size = batch_size
    super().__init__(**kwargs)

  def set_parents(self, parents):
    if len(parents) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = parents[0].out_tensor
    with tf.name_scope(self.name):
      gru_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
      initial_gru_state = gru_cell.zero_state(self.batch_size, tf.float32)
      rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
          gru_cell,
          parent_tensor,
          initial_state=initial_gru_state,
          scope=self.name)
      projection = lambda x: tf.contrib.layers.linear(x, num_outputs=self.out_channels, activation_fn=tf.nn.sigmoid)
      self.out_tensor = tf.map_fn(projection, rnn_outputs)


class TimeSeriesDense(Layer):

  def __init__(self, out_channels, **kwargs):
    self.out_channels = out_channels
    super().__init__(**kwargs)

  def set_parents(self, parents):
    if len(parents) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = parents[0].out_tensor
    with tf.name_scope(self.name):
      dense_fn = lambda x: tf.contrib.layers.fully_connected(x, num_outputs=self.out_channels,
                                                             activation_fn=tf.nn.sigmoid)
      self.out_tensor = tf.map_fn(dense_fn, parent_tensor)


class Input(Layer):

  def __init__(self, t_shape, **kwargs):
    self.t_shape = t_shape
    super().__init__(**kwargs)

  def set_parents(self, parents):
    with tf.name_scope(self.name):
      self.out_tensor = tf.placeholder(tf.float32, shape=self.t_shape)
