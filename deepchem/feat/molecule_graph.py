from typing import Optional, List
import numpy as np


class MoleculeGraphData(object):
  """Molecule Graph Data class for sparse pattern"""

  def __init__(
      self,
      node_features: np.ndarray,
      edge_index: np.ndarray,
      targets: np.ndarray,
      edge_features: Optional[np.ndarray] = None,
      graph_features: Optional[np.ndarray] = None,
  ):
    """

    Parameters
    ----------
    node_features : np.ndarray
      Node feature matrix with shape [num_nodes, num_node_features]
    edge_index : np.ndarray
      Graph connectivity in COO format with shape [2, num_edges]
    targets : np.ndarray
      Graph or node targets with arbitrary shape
    edge_features : np.ndarray, optional (default None)
      Edge feature matrix with shape [num_edges, num_edge_features]
    graph_features : np.ndarray, optional (default None)
      Graph feature vector with shape [num_graph_features,]
    """
    # validate params
    if isinstance(node_features, np.ndarray) is False:
      raise ValueError('node_features must be np.ndarray.')
    if isinstance(edge_index, np.ndarray) is False:
      raise ValueError('edge_index must be np.ndarray.')
    elif edge_index.shape[0] != 2:
      raise ValueError('The shape of edge_index is [2, num_edges].')
    if isinstance(targets, np.ndarray) is False:
      raise ValueError('y must be np.ndarray.')
    if edge_features is not None:
      if isinstance(edge_features, np.ndarray) is False:
        raise ValueError('edge_features must be np.ndarray or None.')
      elif edge_index.shape[1] != edge_features.shape[0]:
        raise ValueError('The first dimension of edge_features must be the \
                    same as the second dimension of edge_index.')
    if graph_features is not None and isinstance(graph_features,
                                                 np.ndarray) is False:
      raise ValueError('graph_features must be np.ndarray or None.')

    self.node_features = node_features
    self.edge_index = edge_index
    self.edge_features = edge_features
    self.graph_features = graph_features
    self.targets = targets
    self.num_nodes, self.num_node_features = self.node_features.shape
    self.num_edges, self.num_edge_features = None, None
    if self.node_features is not None:
      self.num_edges, self.num_edge_features = self.edge_features.shape


class BatchMoleculeGraphData(MoleculeGraphData):
  """Batch Data class for sparse pattern"""

  def __init__(self, molecule_graph_list: List[MoleculeGraphData]):
    """
    Parameters
    ----------
    molecule_graph_list : List[MoleculeGraphData]
      List of MoleculeGraphData
    """
    # stack features and targets
    batch_node_features = np.vstack(
        [graph.node_features for graph in molecule_graph_list])
    batch_targets = np.vstack([graph.targets for graph in molecule_graph_list])

    # before stacking edge_features or graph_features,
    # we should check whether these are None or not
    if molecule_graph_list[0].edge_features is not None:
      batch_edge_features = np.vstack(
          [graph.edge_features for graph in molecule_graph_list])
    else:
      batch_edge_features = None

    if molecule_graph_list[0].graph_features is not None:
      batch_graph_features = np.vstack(
          [graph.graph_features for graph in molecule_graph_list])
    else:
      batch_graph_features = None

    # create new edge index
    num_nodes_list = [graph.num_nodes for graph in molecule_graph_list]
    batch_edge_index = np.hstack(
      [graph.edge_index + prev_num_node for prev_num_node, graph \
        in zip([0] + num_nodes_list[:-1], molecule_graph_list)]
    ).astype(int)

    # graph idx indicates which nodes belong to which graph
    graph_idx = []
    for i, num_nodes in enumerate(num_nodes_list):
      graph_idx.extend([i] * num_nodes)
    self.graph_idx = np.array(graph_idx, dtype=int)

    super().__init__(
        node_features=batch_node_features,
        edge_index=batch_edge_index,
        targets=batch_targets,
        edge_features=batch_edge_features,
        graph_features=batch_graph_features,
    )
