import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, Dense, Concat, SoftMax, SoftMaxCrossEntropy, Layer, \
  GraphConv, BatchNorm, GraphPool, GraphGather, WeightedError
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
import time


class GraphConvTensorGraph(TensorGraph):
  """
  """

  def __init__(self, **kwargs):
    super(GraphConvTensorGraph, self).__init__(**kwargs)
    self.min_degree = 0
    self.max_degree = 10

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
        feed_dict[self.features[i + 3]
                  .out_tensor] = multiConvMol.get_deg_adjacency_lists()[i + 1]
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
              dataset.iterbatches(
                  self.batch_size, deterministic=True, pad_batches=True)):
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
