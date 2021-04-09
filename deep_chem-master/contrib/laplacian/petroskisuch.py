import collections

import numpy as np
import six
import tensorflow as tf

from deepchem.data import NumpyDataset
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveGather, \
  DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, \
  DAGGather, DTNNExtract, MessagePassing, SetGather
from deepchem.models.tensorgraph.graph_layers import WeaveLayerFactory
from deepchem.models.tensorgraph.layers import Dense, SoftMax, \
  SoftMaxCrossEntropy, GraphConv, BatchNorm, \
  GraphPool, GraphGather, WeightedError, Dropout, BatchNorm, Stack, Flatten, GraphCNN, GraphCNNPool
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms


class PetroskiSuchModel(TensorGraph):
  """
      Model from Robust Spatial Filtering with Graph Convolutional Neural Networks
      https://arxiv.org/abs/1703.00792
      """

  def __init__(self,
               n_tasks,
               max_atoms=200,
               dropout=0.0,
               mode="classification",
               **kwargs):
    """
            Parameters
            ----------
            n_tasks: int
              Number of tasks
            mode: str
              Either "classification" or "regression"
            """
    self.n_tasks = n_tasks
    self.mode = mode
    self.max_atoms = max_atoms
    self.error_bars = True if 'error_bars' in kwargs and kwargs['error_bars'] else False
    self.dropout = dropout
    kwargs['use_queue'] = False
    super(PetroskiSuchModel, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    self.vertex_features = Feature(shape=(None, self.max_atoms, 75))
    self.adj_matrix = Feature(shape=(None, self.max_atoms, 1, self.max_atoms))
    self.mask = Feature(shape=(None, self.max_atoms, 1))

    gcnn1 = BatchNorm(
        GraphCNN(
            num_filters=64,
            in_layers=[self.vertex_features, self.adj_matrix, self.mask]))
    gcnn1 = Dropout(self.dropout, in_layers=gcnn1)
    gcnn2 = BatchNorm(
        GraphCNN(num_filters=64, in_layers=[gcnn1, self.adj_matrix, self.mask]))
    gcnn2 = Dropout(self.dropout, in_layers=gcnn2)
    gc_pool, adj_matrix = GraphCNNPool(
        num_vertices=32, in_layers=[gcnn2, self.adj_matrix, self.mask])
    gc_pool = BatchNorm(gc_pool)
    gc_pool = Dropout(self.dropout, in_layers=gc_pool)
    gcnn3 = BatchNorm(GraphCNN(num_filters=32, in_layers=[gc_pool, adj_matrix]))
    gcnn3 = Dropout(self.dropout, in_layers=gcnn3)
    gc_pool2, adj_matrix2 = GraphCNNPool(
        num_vertices=8, in_layers=[gcnn3, adj_matrix])
    gc_pool2 = BatchNorm(gc_pool2)
    gc_pool2 = Dropout(self.dropout, in_layers=gc_pool2)
    flattened = Flatten(in_layers=gc_pool2)
    readout = Dense(
        out_channels=256, activation_fn=tf.nn.relu, in_layers=flattened)
    costs = []
    self.my_labels = []
    for task in range(self.n_tasks):
      if self.mode == 'classification':
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[readout])

        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.my_labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == 'regression':
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[readout])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.my_labels.append(label)
        cost = L2Loss(in_layers=[label, regression])
        costs.append(cost)
    if self.mode == "classification":
      entropy = Stack(in_layers=costs, axis=-1)
    elif self.mode == "regression":
      entropy = Stack(in_layers=costs, axis=1)
    self.my_task_weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[entropy, self.my_task_weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(
              self.batch_size, pad_batches=True, deterministic=deterministic)):
        d = {}
        for index, label in enumerate(self.my_labels):
          if self.mode == 'classification':
            d[label] = to_one_hot(y_b[:, index])
          if self.mode == 'regression':
            d[label] = np.expand_dims(y_b[:, index], -1)
        d[self.my_task_weights] = w_b
        d[self.adj_matrix] = np.expand_dims(np.array([x[0] for x in X_b]), -2)
        d[self.vertex_features] = np.array([x[1] for x in X_b])
        mask = np.zeros(shape=(self.batch_size, self.max_atoms, 1))
        for i in range(self.batch_size):
          mask_size = X_b[i][2]
          mask[i][:mask_size][0] = 1
        d[self.mask] = mask
        yield d

  def predict_proba_on_generator(self, generator, transformers=[]):
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      out_tensors = [x.out_tensor for x in self.outputs]
      results = []
      for feed_dict in generator:
        feed_dict = {
            self.layers[k.name].out_tensor: v
            for k, v in six.iteritems(feed_dict)
        }
        feed_dict[self._training_placeholder] = 1.0  ##
        result = np.array(self.session.run(out_tensors, feed_dict=feed_dict))
        if len(result.shape) == 3:
          result = np.transpose(result, axes=[1, 0, 2])
        if len(transformers) > 0:
          result = undo_transforms(result, transformers)
        results.append(result)
      return np.concatenate(results, axis=0)

  def evaluate(self, dataset, metrics, transformers=[], per_task_metrics=False):
    if not self.built:
      self.build()
    return self.evaluate_generator(
        self.default_generator(dataset, predict=True),
        metrics,
        labels=self.my_labels,
        weights=[self.my_task_weights],
        per_task_metrics=per_task_metrics)
