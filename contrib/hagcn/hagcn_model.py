""" High-Order and Adaptive Graph Convolutional Network (HA-GCN) model, defined in https://arxiv.org/pdf/1706.09916"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import tensorflow as tf

from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature, Label, Weights
from deepchem.models.tensorgraph.layers import Concat
from deepchem.models.tensorgraph.layers import ReduceSum, Dense, ReLU, Flatten, Reshape
from deepchem.models.tensorgraph.layers import L2Loss, WeightedError
from deepchem.feat.mol_graphs import ConvMol
from hagcn_layers import KOrderGraphConv, AdaptiveFilter


class HAGCN(TensorGraph):

  def __init__(self,
               max_nodes,
               num_node_features,
               n_tasks=1,
               k_max=1,
               task_mode='graph',
               combine_method='linear',
               **kwargs):
    """
      Parameters
      ----------
      max_nodes: int
        Maximum number of nodes (atoms) graphs in dataset can have
      num_node_features: int
        Number of features per node
      atoms: list
        List of atoms available across train, valid, test
      k_max: int, optional
        Largest k-hop neighborhood per atom
      batch_size: int, optional
        Batch size used
      task_mode: str, optional
        Whether the model is used for node based tasks or edge based tasks or graph tasks
      combine_method: str, optional
        Combining the inputs for the AdaptiveFilterLayer
    """

    if task_mode not in ['graph', 'node', 'edge']:
      raise ValueError('task_mode must be one of graph, node, edge')

    self.k_max = k_max
    self.n_tasks = n_tasks
    self.max_nodes = max_nodes
    self.num_node_features = num_node_features
    self.task_mode = task_mode
    self.combine_method = combine_method
    super(HAGCN, self).__init__(**kwargs)

    self._build()

  def _build(self):
    self.A_tilda_k = list()
    for k in range(1, self.k_max + 1):
      self.A_tilda_k.append(
          Feature(
              name="graph_adjacency_{}".format(k),
              dtype=tf.float32,
              shape=[None, self.max_nodes, self.max_nodes]))
    self.X = Feature(
        name='atom_features',
        dtype=tf.float32,
        shape=[None, self.max_nodes, self.num_node_features])

    graph_layers = list()
    adaptive_filters = list()

    for index, k in enumerate(range(1, self.k_max + 1)):

      in_layers = [self.A_tilda_k[index], self.X]

      adaptive_filters.append(
          AdaptiveFilter(
              batch_size=self.batch_size,
              in_layers=in_layers,
              num_nodes=self.max_nodes,
              num_node_features=self.num_node_features,
              combine_method=self.combine_method))

      graph_layers.append(
          KOrderGraphConv(
              batch_size=self.batch_size,
              in_layers=in_layers + [adaptive_filters[index]],
              num_nodes=self.max_nodes,
              num_node_features=self.num_node_features,
              init='glorot_uniform'))

    graph_features = Concat(in_layers=graph_layers, axis=2)
    graph_features = ReLU(in_layers=[graph_features])
    flattened = Flatten(in_layers=[graph_features])

    dense1 = Dense(
        in_layers=[flattened], out_channels=64, activation_fn=tf.nn.relu)
    dense2 = Dense(
        in_layers=[dense1], out_channels=16, activation_fn=tf.nn.relu)
    dense3 = Dense(
        in_layers=[dense2], out_channels=1 * self.n_tasks, activation_fn=None)
    output = Reshape(in_layers=[dense3], shape=(-1, self.n_tasks, 1))
    self.add_output(output)

    label = Label(shape=(None, self.n_tasks, 1))
    weights = Weights(shape=(None, self.n_tasks))
    loss = ReduceSum(L2Loss(in_layers=[label, output]))

    weighted_loss = WeightedError(in_layers=[loss, weights])
    self.set_loss(weighted_loss)

  @staticmethod
  def pow_k(inputs, k=1):
    """Computes the kth power of inputs, used for adjacency matrix"""
    if k == 1:
      return inputs
    if k == 0:
      return np.ones(inputs.shape)

    if k % 2 == 0:
      half = HAGCN.pow_k(inputs, k=k // 2)
      return np.matmul(half, half)
    else:
      return np.matmul(inputs, HAGCN.pow_k(inputs, (k - 1) // 2))

  def compute_adjacency_matrix(self, mol):
    """Computes the adjacency matrix for a mol."""
    assert isinstance(mol, ConvMol)
    canon_adj_lists = mol.get_adjacency_list()
    adjacency = np.zeros((self.max_nodes, self.max_nodes))
    for atom_idx, connections in enumerate(canon_adj_lists):
      for neighbor_idx in connections:
        adjacency[atom_idx, neighbor_idx] = 1
    return adjacency

  @staticmethod
  def compute_a_tilda_k(inputs, k=1):
    A_k = HAGCN.pow_k(inputs, k)
    A_k_I = A_k + np.eye(inputs.shape[-1])
    A_tilda_k = np.minimum(A_k_I, 1)
    return A_tilda_k

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        feed_dict = {}
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        if y_b is not None:
          feed_dict[self.labels[0]] = y_b

        atom_features = list()
        A_tilda_k = [[] for _ in range(1, self.k_max + 1)]

        for im, mol in enumerate(X_b):
          # Atom features with padding
          num_atoms = mol.get_num_atoms()
          atom_feats = mol.get_atom_features()
          num_to_pad = self.max_nodes - num_atoms
          if num_to_pad > 0:
            to_pad = np.zeros((num_to_pad, self.num_node_features))
            atom_feats = np.concatenate([atom_feats, to_pad], axis=0)
          atom_features.append(atom_feats)

          # A_tilda_k computation
          adjacency = self.compute_adjacency_matrix(mol)
          for i, k in enumerate(range(1, self.k_max + 1)):
            A_tilda_k[i].append(HAGCN.compute_a_tilda_k(adjacency, k=k))

        # Final feed_dict setup
        atom_features = np.asarray(atom_features)
        for i, k in enumerate(range(1, self.k_max + 1)):
          val = np.asarray(A_tilda_k[i])
          # assert val.shape == (self.batch_size, self.max_nodes, self.max_nodes)
          feed_dict[self.A_tilda_k[i]] = val
        #assert atom_features.shape == (self.batch_size, self.max_nodes,
        #                               self.num_node_features)
        feed_dict[self.X] = atom_features

        yield feed_dict
