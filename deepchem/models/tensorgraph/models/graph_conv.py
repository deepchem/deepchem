import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, Dense, Concat, SoftMax, SoftMaxCrossEntropy, Layer, \
  GraphConvLayer, BatchNormLayer, GraphPoolLayer, GraphGather, WeightedError
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
import time



class GraphConvTensorGraph(TensorGraph):
  """
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.min_degree=0
    self.max_degree=10

  def _construct_feed_dict(self, X_b, y_b, w_b, ids_b):
    feed_dict = dict()
    if y_b is not None:
      for index, label in enumerate(self.labels):
        feed_dict[label.out_tensor] = to_one_hot(y_b[:, index])
    if self.task_weights is not None and w_b is not None:
      feed_dict[self.task_weights[0].out_tensor] = w_b
    if self.features is not None:
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      feed_dict[self.features[0].out_tensor] = multiConvMol.get_atom_features()
      feed_dict[self.features[1].out_tensor] = multiConvMol.deg_slice
      feed_dict[self.features[2].out_tensor] = multiConvMol.membership
      for i in range(self.max_degree):
        feed_dict[self.features[i + 3].out_tensor] = multiConvMol.get_deg_adjacency_lists()[i + 1]
    return feed_dict

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          checkpoint_interval=10):
    """
    TODO(LESWING) put this logic into tensor_graph or figure out how to use an input queue.
    Parameters
    ----------
    dataset
    nb_epoch
    max_checkpoints_to_keep
    log_every_N_batches
    checkpoint_interval

    Returns
    -------

    """
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      time1 = time.time()
      train_op = self._get_tf('train_op')
      saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      with tf.Session() as sess:
        self._initialize_weights(sess, saver)
        avg_loss, n_batches = 0.0, 0.0
        for epoch in range(nb_epoch):
          for ind, (X_b, y_b, w_b, ids_b) in enumerate(
            dataset.iterbatches(self.batch_size, deterministic=True, pad_batches=True)):
            feed_dict = self._construct_feed_dict(X_b, y_b, w_b, ids_b)
            output_tensors = [x.out_tensor for x in self.outputs]
            fetches = output_tensors + [train_op, self.loss.out_tensor]
            fetched_values = sess.run(fetches, feed_dict=feed_dict)
            loss = fetched_values[-1]
            avg_loss += loss
            n_batches += 1
            self.global_step += 1
          if epoch % checkpoint_interval == checkpoint_interval - 1:
            saver.save(sess, self.save_file, global_step=self.global_step)
            avg_loss = float(avg_loss) / n_batches
            print('Ending epoch %d: Average loss %g' % (epoch, avg_loss))
        saver.save(sess, self.save_file, global_step=self.global_step)
        self.last_checkpoint = saver.last_checkpoints[-1]
      ############################################################## TIMING
      time2 = time.time()
      print("TIMING: model fitting took %0.3f s" % (time2 - time1))
      ############################################################## TIMING


def graph_conv_model(batch_size, num_tasks):
  model = GraphConvTensorGraph(batch_size=batch_size,
                               use_queue=False)
  atom_features = Input(shape=(None, 75))
  model.add_feature(atom_features)

  degree_slice = Input(shape=(None, 2), dtype=tf.int32)
  model.add_feature(degree_slice)

  membership = Input(shape=(None,), dtype=tf.int32)
  model.add_feature(membership)

  deg_adjs = []
  for i in range(model.min_degree, model.max_degree + 1):
    deg_adj = Input(shape=(None, i + 1), dtype=tf.int32)
    model.add_feature(deg_adj)
    deg_adjs.append(deg_adj)

  gc1 = GraphConvLayer(64, activation_fn=tf.nn.relu)
  model.add_layer(gc1, parents=[atom_features, degree_slice, membership] + deg_adjs)

  batch_norm1 = BatchNormLayer()
  model.add_layer(batch_norm1, parents=[gc1])

  gp1 = GraphPoolLayer()
  model.add_layer(gp1, parents=[batch_norm1, degree_slice, membership] + deg_adjs)

  gc2 = GraphConvLayer(64, activation_fn=tf.nn.relu)
  model.add_layer(gc2, parents=[gp1, degree_slice, membership] + deg_adjs)

  batch_norm2 = BatchNormLayer()
  model.add_layer(batch_norm2, parents=[gc2])

  gp2 = GraphPoolLayer()
  model.add_layer(gp2, parents=[batch_norm2, degree_slice, membership] + deg_adjs)

  dense = Dense(out_channels=128, activation_fn=None)
  model.add_layer(dense, parents=[gp2])

  batch_norm3 = BatchNormLayer()
  model.add_layer(batch_norm3, parents=[dense])

  gg1 = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)
  model.add_layer(gg1, parents=[batch_norm3, degree_slice, membership] + deg_adjs)

  costs = []
  for task in range(num_tasks):
    classification = Dense(out_channels=2, name="GUESS%s" % task, activation_fn=None)
    model.add_layer(classification, parents=[gg1])

    softmax = SoftMax(name="SOFTMAX%s" % task)
    model.add_layer(softmax, parents=[classification])
    model.add_output(softmax)

    label = Input(shape=(None, 2), name="LABEL%s" % task)
    model.add_label(label)

    cost = SoftMaxCrossEntropy(name="COST%s" % task)
    model.add_layer(cost, parents=[label, classification])
    costs.append(cost)

  entropy = Concat(name="ENT")
  model.add_layer(entropy, parents=costs)

  task_weights = Input(shape=(None, num_tasks), name="W")
  model.add_task_weight(task_weights)

  loss = WeightedError(name="ERROR")
  model.add_layer(loss, parents=[entropy, task_weights])
  model.set_loss(loss)
  return model
