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
from deepchem.models.tensorgraph.layers import Dense, Concat, SoftMax, \
  SoftMaxCrossEntropy, GraphConv, BatchNorm, \
  GraphPool, GraphGather, WeightedError, Dropout, BatchNormalization, Stack, Flatten, GraphCNN, GraphCNNPool
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms


class WeaveTensorGraph(TensorGraph):

  def __init__(self,
               n_tasks,
               n_atom_feat=75,
               n_pair_feat=14,
               n_hidden=50,
               n_graph_feat=128,
               mode="classification",
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_atom_feat: int, optional
      Number of features per atom.
    n_pair_feat: int, optional
      Number of features per pair of atoms.
    n_hidden: int, optional
      Number of units(convolution depths) in corresponding hidden layer
    n_graph_feat: int, optional
      Number of output features for each molecule(graph)
    mode: str
      Either "classification" or "regression" for type of model.
    """
    self.n_tasks = n_tasks
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.n_graph_feat = n_graph_feat
    self.mode = mode
    super(WeaveTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    """Building graph structures:
        Features => WeaveLayer => WeaveLayer => Dense => WeaveGather => Classification or Regression
        """
    self.atom_features = Feature(shape=(None, self.n_atom_feat))
    self.pair_features = Feature(shape=(None, self.n_pair_feat))
    self.pair_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_to_pair = Feature(shape=(None, 2), dtype=tf.int32)
    weave_layer1A, weave_layer1P = WeaveLayerFactory(
        n_atom_input_feat=self.n_atom_feat,
        n_pair_input_feat=self.n_pair_feat,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden,
        in_layers=[
            self.atom_features, self.pair_features, self.pair_split,
            self.atom_to_pair
        ])
    weave_layer2A, weave_layer2P = WeaveLayerFactory(
        n_atom_input_feat=self.n_hidden,
        n_pair_input_feat=self.n_hidden,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden,
        update_pair=False,
        in_layers=[
            weave_layer1A, weave_layer1P, self.pair_split, self.atom_to_pair
        ])
    dense1 = Dense(
        out_channels=self.n_graph_feat,
        activation_fn=tf.nn.tanh,
        in_layers=weave_layer2A)
    batch_norm1 = BatchNormalization(epsilon=1e-5, mode=1, in_layers=[dense1])
    weave_gather = WeaveGather(
        self.batch_size,
        n_input=self.n_graph_feat,
        gaussian_expand=True,
        in_layers=[batch_norm1, self.atom_split])

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      if self.mode == "classification":
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[weave_gather])
        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.labels_fd.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == "regression":
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[weave_gather])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.labels_fd.append(label)
        cost = L2Loss(in_layers=[label, regression])
        costs.append(cost)
    if self.mode == "classification":
      all_cost = Concat(in_layers=costs, axis=1)
    elif self.mode == "regression":
      all_cost = Stack(in_layers=costs, axis=1)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    """ TensorGraph style implementation
    similar to deepchem.models.tf_new_models.graph_topology.AlternateWeaveTopology.batch_to_feed_dict
    """
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            if self.mode == "classification":
              feed_dict[label] = to_one_hot(y_b[:, index])
            if self.mode == "regression":
              feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None:
          feed_dict[self.weights] = w_b

        atom_feat = []
        pair_feat = []
        atom_split = []
        atom_to_pair = []
        pair_split = []
        start = 0
        for im, mol in enumerate(X_b):
          n_atoms = mol.get_num_atoms()
          # number of atoms in each molecule
          atom_split.extend([im] * n_atoms)
          # index of pair features
          C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
          atom_to_pair.append(
              np.transpose(
                  np.array([C1.flatten() + start,
                            C0.flatten() + start])))
          # number of pairs for each atom
          pair_split.extend(C1.flatten() + start)
          start = start + n_atoms

          # atom features
          atom_feat.append(mol.get_atom_features())
          # pair features
          pair_feat.append(
              np.reshape(mol.get_pair_features(), (n_atoms * n_atoms,
                                                   self.n_pair_feat)))

        feed_dict[self.atom_features] = np.concatenate(atom_feat, axis=0)
        feed_dict[self.pair_features] = np.concatenate(pair_feat, axis=0)
        feed_dict[self.pair_split] = np.array(pair_split)
        feed_dict[self.atom_split] = np.array(atom_split)
        feed_dict[self.atom_to_pair] = np.concatenate(atom_to_pair, axis=0)
        yield feed_dict


class DTNNTensorGraph(TensorGraph):

  def __init__(self,
               n_tasks,
               n_embedding=30,
               n_hidden=100,
               n_distance=100,
               distance_min=-1,
               distance_max=18,
               output_activation=True,
               mode="classification",
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_embedding: int, optional
      Number of features per atom.
    n_hidden: int, optional
      Number of features for each molecule after DTNNStep
    n_distance: int, optional
      granularity of distance matrix
      step size will be (distance_max-distance_min)/n_distance
    distance_min: float, optional
      minimum distance of atom pairs, default = -1 Angstorm
    distance_max: float, optional
      maximum distance of atom pairs, default = 18 Angstorm
    mode: str
      Either "classification" or "regression" for type of model.
    """
    self.n_tasks = n_tasks
    self.n_embedding = n_embedding
    self.n_hidden = n_hidden
    self.n_distance = n_distance
    self.distance_min = distance_min
    self.distance_max = distance_max
    self.step_size = (distance_max - distance_min) / n_distance
    self.steps = np.array(
        [distance_min + i * self.step_size for i in range(n_distance)])
    self.steps = np.expand_dims(self.steps, 0)
    self.output_activation = output_activation
    self.mode = mode
    super(DTNNTensorGraph, self).__init__(**kwargs)
    assert self.mode == "regression"
    self.build_graph()

  def build_graph(self):
    """Building graph structures:
    Features => DTNNEmbedding => DTNNStep => DTNNStep => DTNNGather => Regression
    """
    self.atom_number = Feature(shape=(None,), dtype=tf.int32)
    self.distance = Feature(shape=(None, self.n_distance))
    self.atom_membership = Feature(shape=(None,), dtype=tf.int32)
    self.distance_membership_i = Feature(shape=(None,), dtype=tf.int32)
    self.distance_membership_j = Feature(shape=(None,), dtype=tf.int32)

    dtnn_embedding = DTNNEmbedding(
        n_embedding=self.n_embedding, in_layers=[self.atom_number])
    dtnn_layer1 = DTNNStep(
        n_embedding=self.n_embedding,
        n_distance=self.n_distance,
        in_layers=[
            dtnn_embedding, self.distance, self.distance_membership_i,
            self.distance_membership_j
        ])
    dtnn_layer2 = DTNNStep(
        n_embedding=self.n_embedding,
        n_distance=self.n_distance,
        in_layers=[
            dtnn_layer1, self.distance, self.distance_membership_i,
            self.distance_membership_j
        ])
    dtnn_gather = DTNNGather(
        n_embedding=self.n_embedding,
        layer_sizes=[self.n_hidden],
        n_outputs=self.n_tasks,
        output_activation=self.output_activation,
        in_layers=[dtnn_layer2, self.atom_membership])

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      regression = DTNNExtract(task, in_layers=[dtnn_gather])
      self.add_output(regression)
      label = Label(shape=(None, 1))
      self.labels_fd.append(label)
      cost = L2Loss(in_layers=[label, regression])
      costs.append(cost)

    all_cost = Stack(in_layers=costs, axis=1)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    """ TensorGraph style implementation
        similar to deepchem.models.tf_new_models.graph_topology.DTNNGraphTopology.batch_to_feed_dict
        """
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None:
          feed_dict[self.weights] = w_b
        distance = []
        atom_membership = []
        distance_membership_i = []
        distance_membership_j = []
        num_atoms = list(map(sum, X_b.astype(bool)[:, :, 0]))
        atom_number = [
            np.round(
                np.power(2 * np.diag(X_b[i, :num_atoms[i], :num_atoms[i]]), 1 /
                         2.4)).astype(int) for i in range(len(num_atoms))
        ]
        start = 0
        for im, molecule in enumerate(atom_number):
          distance_matrix = np.outer(
              molecule, molecule) / X_b[im, :num_atoms[im], :num_atoms[im]]
          np.fill_diagonal(distance_matrix, -100)
          distance.append(np.expand_dims(distance_matrix.flatten(), 1))
          atom_membership.append([im] * num_atoms[im])
          membership = np.array([np.arange(num_atoms[im])] * num_atoms[im])
          membership_i = membership.flatten(order='F')
          membership_j = membership.flatten()
          distance_membership_i.append(membership_i + start)
          distance_membership_j.append(membership_j + start)
          start = start + num_atoms[im]
        feed_dict[self.atom_number] = np.concatenate(atom_number)
        distance = np.concatenate(distance, 0)
        feed_dict[self.distance] = np.exp(-np.square(distance - self.steps) /
                                          (2 * self.step_size**2))
        feed_dict[self.distance_membership_i] = np.concatenate(
            distance_membership_i)
        feed_dict[self.distance_membership_j] = np.concatenate(
            distance_membership_j)
        feed_dict[self.atom_membership] = np.concatenate(atom_membership)

        yield feed_dict


class DAGTensorGraph(TensorGraph):

  def __init__(self,
               n_tasks,
               max_atoms=50,
               n_atom_feat=75,
               n_graph_feat=30,
               n_outputs=30,
               mode="classification",
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    max_atoms: int, optional
      Maximum number of atoms in a molecule, should be defined based on dataset
    n_atom_feat: int, optional
      Number of features per atom.
    n_graph_feat: int, optional
      Number of features for atom in the graph
    n_outputs: int, optional
      Number of features for each molecule
    mode: str
      Either "classification" or "regression" for type of model.
    """
    self.n_tasks = n_tasks
    self.max_atoms = max_atoms
    self.n_atom_feat = n_atom_feat
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs
    self.mode = mode
    super(DAGTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    """Building graph structures:
        Features => DAGLayer => DAGGather => Classification or Regression
        """
    self.atom_features = Feature(shape=(None, self.n_atom_feat))
    self.parents = Feature(
        shape=(None, self.max_atoms, self.max_atoms), dtype=tf.int32)
    self.calculation_orders = Feature(
        shape=(None, self.max_atoms), dtype=tf.int32)
    self.calculation_masks = Feature(
        shape=(None, self.max_atoms), dtype=tf.bool)
    self.membership = Feature(shape=(None,), dtype=tf.int32)
    self.n_atoms = Feature(shape=(), dtype=tf.int32)
    dag_layer1 = DAGLayer(
        n_graph_feat=self.n_graph_feat,
        n_atom_feat=self.n_atom_feat,
        max_atoms=self.max_atoms,
        batch_size=self.batch_size,
        in_layers=[
            self.atom_features, self.parents, self.calculation_orders,
            self.calculation_masks, self.n_atoms
        ])
    dag_gather = DAGGather(
        n_graph_feat=self.n_graph_feat,
        n_outputs=self.n_outputs,
        max_atoms=self.max_atoms,
        in_layers=[dag_layer1, self.membership])

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      if self.mode == "classification":
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[dag_gather])
        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.labels_fd.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == "regression":
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[dag_gather])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.labels_fd.append(label)
        cost = L2Loss(in_layers=[label, regression])
        costs.append(cost)
    if self.mode == "classification":
      all_cost = Concat(in_layers=costs, axis=1)
    elif self.mode == "regression":
      all_cost = Stack(in_layers=costs, axis=1)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    """ TensorGraph style implementation
        similar to deepchem.models.tf_new_models.graph_topology.DAGGraphTopology.batch_to_feed_dict
        """
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            if self.mode == "classification":
              feed_dict[label] = to_one_hot(y_b[:, index])
            if self.mode == "regression":
              feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None:
          feed_dict[self.weights] = w_b

        atoms_per_mol = [mol.get_num_atoms() for mol in X_b]
        n_atoms = sum(atoms_per_mol)
        start_index = [0] + list(np.cumsum(atoms_per_mol)[:-1])

        atoms_all = []
        # calculation orders for a batch of molecules
        parents_all = []
        calculation_orders = []
        calculation_masks = []
        membership = []
        for idm, mol in enumerate(X_b):
          # padding atom features vector of each molecule with 0
          atoms_all.append(mol.get_atom_features())
          parents = mol.parents
          parents_all.extend(parents)
          calculation_index = np.array(parents)[:, :, 0]
          mask = np.array(calculation_index - self.max_atoms, dtype=bool)
          calculation_orders.append(calculation_index + start_index[idm])
          calculation_masks.append(mask)
          membership.extend([idm] * atoms_per_mol[idm])

        feed_dict[self.atom_features] = np.concatenate(atoms_all, axis=0)
        feed_dict[self.parents] = np.stack(parents_all, axis=0)
        feed_dict[self.calculation_orders] = np.concatenate(
            calculation_orders, axis=0)
        feed_dict[self.calculation_masks] = np.concatenate(
            calculation_masks, axis=0)
        feed_dict[self.membership] = np.array(membership)
        feed_dict[self.n_atoms] = n_atoms
        yield feed_dict


class PetroskiSuchTensorGraph(TensorGraph):
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
    super(PetroskiSuchTensorGraph, self).__init__(**kwargs)
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
      entropy = Concat(in_layers=costs, axis=-1)
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


class GraphConvTensorGraph(TensorGraph):

  def __init__(self, n_tasks, mode="classification", **kwargs):
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
    self.error_bars = True if 'error_bars' in kwargs and kwargs['error_bars'] else False
    kwargs['use_queue'] = False
    super(GraphConvTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    """
    Building graph structures:
    """
    self.atom_features = Feature(shape=(None, 75))
    self.degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
    self.membership = Feature(shape=(None,), dtype=tf.int32)

    self.deg_adjs = []
    for i in range(0, 10 + 1):
      deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
      self.deg_adjs.append(deg_adj)
    gc1 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[self.atom_features, self.degree_slice, self.membership] +
        self.deg_adjs)
    batch_norm1 = BatchNorm(in_layers=[gc1])
    gp1 = GraphPool(in_layers=[batch_norm1, self.degree_slice, self.membership]
                    + self.deg_adjs)
    gc2 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[gp1, self.degree_slice, self.membership] + self.deg_adjs)
    batch_norm2 = BatchNorm(in_layers=[gc2])
    gp2 = GraphPool(in_layers=[batch_norm2, self.degree_slice, self.membership]
                    + self.deg_adjs)
    dense = Dense(out_channels=128, activation_fn=tf.nn.relu, in_layers=[gp2])
    batch_norm3 = BatchNorm(in_layers=[dense])
    readout = GraphGather(
        batch_size=self.batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[batch_norm3, self.degree_slice, self.membership] +
        self.deg_adjs)

    if self.error_bars == True:
      readout = Dropout(in_layers=[readout], dropout_prob=0.2)

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
      entropy = Concat(in_layers=costs, axis=-1)
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
              self.batch_size,
              pad_batches=pad_batches,
              deterministic=deterministic)):
        d = {}
        for index, label in enumerate(self.my_labels):
          if self.mode == 'classification':
            d[label] = to_one_hot(y_b[:, index])
          if self.mode == 'regression':
            d[label] = np.expand_dims(y_b[:, index], -1)
        d[self.my_task_weights] = w_b
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        d[self.atom_features] = multiConvMol.get_atom_features()
        d[self.degree_slice] = multiConvMol.deg_slice
        d[self.membership] = multiConvMol.membership
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          d[self.deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
        yield d

  def predict_on_generator(self, generator, transformers=[], outputs=None):
    if not self.built:
      self.build()
    if outputs is None:
      outputs = self.outputs
    elif not isinstance(outputs, collections.Sequence):
      outputs = [outputs]
    with self._get_tf("Graph").as_default():
      # Gather results for each output
      results = [[] for out in outputs]
      for feed_dict in generator:
        feed_dict = {
            self.layers[k.name].out_tensor: v
            for k, v in six.iteritems(feed_dict)
        }
        # Recording the number of samples in the input batch
        n_samples = max(feed_dict[self.membership.out_tensor]) + 1
        feed_dict[self._training_placeholder] = 0.0
        feed_results = self.session.run(outputs, feed_dict=feed_dict)
        if len(feed_results) > 1:
          if len(transformers):
            raise ValueError("Does not support transformations "
                             "for multiple outputs.")
        elif len(feed_results) == 1:
          result = undo_transforms(feed_results[0], transformers)
          feed_results = [result]
        for ind, result in enumerate(feed_results):
          # GraphConvTensorGraph constantly outputs batch_size number of
          # results, only valid samples should be appended to final results
          results[ind].append(result[:n_samples])

      final_results = []
      for result_list in results:
        final_results.append(np.concatenate(result_list, axis=0))
      # If only one output, just return array
      if len(final_results) == 1:
        return final_results[0]
      else:
        return final_results

  def predict_proba_on_generator(self, generator, transformers=[],
                                 outputs=None):
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
        n_samples = max(feed_dict[self.membership.out_tensor]) + 1
        feed_dict[self._training_placeholder] = 1.0  ##
        result = np.array(self.session.run(out_tensors, feed_dict=feed_dict))
        if len(result.shape) == 3:
          result = np.transpose(result, axes=[1, 0, 2])
        if len(transformers) > 0:
          result = undo_transforms(result, transformers)
        results.append(result[:n_samples])
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

  def bayesian_predict(self,
                       dataset,
                       transformers=[],
                       n_passes=4,
                       untransform=False):
    """Generates predictions and confidences on a dataset object
     https://arxiv.org/pdf/1506.02142.pdf

    # Returns:
      mu: numpy ndarray of shape (n_samples, n_tasks)
      sigma: numpy ndarray of shape (n_samples, n_tasks)
    """
    X = dataset.X
    max_index = X.shape[0] - 1
    num_batches = (max_index // self.batch_size) + 1

    mus = []
    sigmas = []
    for i in range(num_batches):
      start = i * self.batch_size
      end = min((i + 1) * self.batch_size, max_index + 1)
      batch = X[start:end]
      mu, sigma = self.bayesian_predict_on_batch(
          batch, transformers=[], n_passes=n_passes)
      mus.append(mu)
      sigmas.append(sigma)
    mu = np.concatenate(mus, axis=0)
    sigma = np.concatenate(sigmas, axis=0) + 0.55

    if untransform:
      mu = undo_transforms(mu, transformers)
      for i in range(sigma.shape[1]):
        sigma[:, i] = sigma[:, i] * transformers[0].y_stds[i]

    return mu[:max_index + 1], sigma[:max_index + 1]

  def bayesian_predict_on_batch(self, X, transformers=[], n_passes=4):
    """
    Returns:
      mu: numpy ndarray of shape (n_samples, n_tasks)
      sigma: numpy ndarray of shape (n_samples, n_tasks)
    """
    dataset = NumpyDataset(X=X, y=None, n_tasks=len(self.outputs))
    y_ = []
    for i in range(n_passes):
      generator = self.default_generator(
          dataset, predict=True, pad_batches=True)
      y_.append(self.predict_on_generator(generator, transformers))

    # Concatenates along 0-th dimension
    y_ = np.array(y_)
    mu = np.mean(y_, axis=0)
    sigma = np.std(y_, axis=0)

    return mu, sigma

  def predict_on_smiles(self, smiles, transformers=[], untransform=False):
    """Generates predictions on a numpy array of smile strings

    # Returns:
      y_: numpy ndarray of shape (n_samples, n_tasks)
    """
    max_index = len(smiles) - 1
    n_tasks = len(self.outputs)
    num_batches = (max_index // self.batch_size) + 1
    featurizer = ConvMolFeaturizer()

    y_ = []
    for i in range(num_batches):
      start = i * self.batch_size
      end = min((i + 1) * self.batch_size, max_index + 1)
      smiles_batch = smiles[start:end]
      y_.append(
          self.predict_on_smiles_batch(smiles_batch, featurizer, transformers))
    y_ = np.concatenate(y_, axis=0)[:max_index + 1]
    y_ = y_.reshape(-1, n_tasks)

    if untransform:
      y_ = undo_transforms(y_, transformers)

    return y_


class MPNNTensorGraph(TensorGraph):
  """ Message Passing Neural Network,
  default structures built according to https://arxiv.org/abs/1511.06391 """

  def __init__(self,
               n_tasks,
               n_atom_feat=70,
               n_pair_feat=8,
               n_hidden=100,
               T=5,
               M=10,
               mode="regression",
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_atom_feat: int, optional
      Number of features per atom.
    n_pair_feat: int, optional
      Number of features per pair of atoms.
    n_hidden: int, optional
      Number of units(convolution depths) in corresponding hidden layer
    n_graph_feat: int, optional
      Number of output features for each molecule(graph)

    """
    self.n_tasks = n_tasks
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.T = T
    self.M = M
    self.mode = mode
    super(MPNNTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    # Build placeholders
    self.atom_features = Feature(shape=(None, self.n_atom_feat))
    self.pair_features = Feature(shape=(None, self.n_pair_feat))
    self.atom_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_to_pair = Feature(shape=(None, 2), dtype=tf.int32)

    message_passing = MessagePassing(
        self.T,
        message_fn='enn',
        update_fn='gru',
        n_hidden=self.n_hidden,
        in_layers=[self.atom_features, self.pair_features, self.atom_to_pair])

    atom_embeddings = Dense(self.n_hidden, in_layers=[message_passing])

    mol_embeddings = SetGather(
        self.M,
        self.batch_size,
        n_hidden=self.n_hidden,
        in_layers=[atom_embeddings, self.atom_split])

    dense1 = Dense(
        out_channels=2 * self.n_hidden,
        activation_fn=tf.nn.relu,
        in_layers=[mol_embeddings])
    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      if self.mode == "classification":
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[dense1])
        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.labels_fd.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == "regression":
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[dense1])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.labels_fd.append(label)
        cost = L2Loss(in_layers=[label, regression])
        costs.append(cost)
    if self.mode == "classification":
      all_cost = Concat(in_layers=costs, axis=1)
    elif self.mode == "regression":
      all_cost = Stack(in_layers=costs, axis=1)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    """ Same generator as Weave models """
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            if self.mode == "classification":
              feed_dict[label] = to_one_hot(y_b[:, index])
            if self.mode == "regression":
              feed_dict[label] = y_b[:, index:index + 1]
        # w_b act as the indicator of unique samples in the batch
        if w_b is not None:
          feed_dict[self.weights] = w_b

        atom_feat = []
        pair_feat = []
        atom_split = []
        atom_to_pair = []
        pair_split = []
        start = 0
        for im, mol in enumerate(X_b):
          n_atoms = mol.get_num_atoms()
          # number of atoms in each molecule
          atom_split.extend([im] * n_atoms)
          # index of pair features
          C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
          atom_to_pair.append(
              np.transpose(
                  np.array([C1.flatten() + start,
                            C0.flatten() + start])))
          # number of pairs for each atom
          pair_split.extend(C1.flatten() + start)
          start = start + n_atoms

          # atom features
          atom_feat.append(mol.get_atom_features())
          # pair features
          pair_feat.append(
              np.reshape(mol.get_pair_features(), (n_atoms * n_atoms,
                                                   self.n_pair_feat)))

        feed_dict[self.atom_features] = np.concatenate(atom_feat, axis=0)
        feed_dict[self.pair_features] = np.concatenate(pair_feat, axis=0)
        feed_dict[self.atom_split] = np.array(atom_split)
        feed_dict[self.atom_to_pair] = np.concatenate(atom_to_pair, axis=0)
        yield feed_dict

  def predict(self, dataset, transformers=[], batch_size=None):
    # MPNN only accept padded input
    generator = self.default_generator(dataset, predict=True, pad_batches=True)
    return self.predict_on_generator(generator, transformers)

  def predict_proba(self, dataset, transformers=[], batch_size=None):
    # MPNN only accept padded input
    generator = self.default_generator(dataset, predict=True, pad_batches=True)
    return self.predict_proba_on_generator(generator, transformers)

  def predict_proba_on_generator(self, generator, transformers=[]):
    """
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      out_tensors = [x.out_tensor for x in self.outputs]
      results = []
      for feed_dict in generator:
        # Extract number of unique samples in the batch from w_b
        n_valid_samples = len(np.nonzero(feed_dict[self.weights][:, 0])[0])
        feed_dict = {
            self.layers[k.name].out_tensor: v
            for k, v in six.iteritems(feed_dict)
        }
        feed_dict[self._training_placeholder] = 0.0
        result = np.array(self.session.run(out_tensors, feed_dict=feed_dict))
        if len(result.shape) == 3:
          result = np.transpose(result, axes=[1, 0, 2])
        result = undo_transforms(result, transformers)
        # Only fetch the first set of unique samples
        results.append(result[:n_valid_samples])
      return np.concatenate(results, axis=0)

  def predict_on_generator(self, generator, transformers=[]):
    return self.predict_proba_on_generator(generator, transformers)