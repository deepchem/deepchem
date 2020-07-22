from typing import Optional, Sequence
import numpy as np


class MoleculeGraphData:
  """MoleculeGraphData class

  This data class is almost same as `torch_geometric.data.Data 
  <https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data>`_.

  Attributes
  ----------
  node_features : np.ndarray
    Node feature matrix with shape [num_nodes, num_node_features]
  edge_index : np.ndarray, dtype int
    Graph connectivity in COO format with shape [2, num_edges]
  targets : np.ndarray
    Graph or node targets with arbitrary shape
  edge_features : np.ndarray, optional (default None)
    Edge feature matrix with shape [num_edges, num_edge_features]
  graph_features : np.ndarray, optional (default None)
    Graph feature vector with shape [num_graph_features,]
  num_nodes : int
    The number of nodes in the graph
  num_node_features : int
    The number of features per node in the graph
  num_edges : int
    The number of edges in the graph
  num_edges_features : int, , optional (default None)
    The number of features per edge in the graph
  """

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
    edge_index : np.ndarray, dtype int
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
    elif edge_index.dtype != np.int:
      raise ValueError('edge_index.dtype must be np.int')
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
    self.num_edges = edge_index.shape[1]
    if self.node_features is not None:
      self.num_edge_features = self.edge_features.shape[1]

  def to_pyg_data(self):
    """Convert to PyTorch Geometric Data instance

    Returns
    -------
    torch_geometric.data.Data
      Molecule graph data for PyTorch Geometric
    """
    try:
      import torch
      from torch_geometric.data import Data
    except ModuleNotFoundError:
      raise ValueError("This class requires PyTorch Geometric to be installed.")

    return Data(
      x=torch.from_numpy(self.node_features),
      edge_index=torch.from_numpy(self.edge_index),
      edge_attr=None if self.edge_features is None \
        else torch.from_numpy(self.edge_features),
      y=torch.from_numpy(self.targets),
    )


class BatchMoleculeGraphData(MoleculeGraphData):
  """Batch MoleculeGraphData class
  
  Attributes
  ----------
  graph_index : np.ndarray, dtype int
    This vector indicates which graph the node belongs with shape [num_nodes,]
  """

  def __init__(self, molecule_graphs: Sequence[MoleculeGraphData]):
    """
    Parameters
    ----------
    molecule_graphs : Sequence[MoleculeGraphData]
      List of MoleculeGraphData
    """
    # stack features and targets
    batch_node_features = np.vstack(
        [graph.node_features for graph in molecule_graphs])
    batch_targets = np.vstack([graph.targets for graph in molecule_graphs])

    # before stacking edge_features or graph_features,
    # we should check whether these are None or not
    if molecule_graphs[0].edge_features is not None:
      batch_edge_features = np.vstack(
          [graph.edge_features for graph in molecule_graphs])
    else:
      batch_edge_features = None

    if molecule_graphs[0].graph_features is not None:
      batch_graph_features = np.vstack(
          [graph.graph_features for graph in molecule_graphs])
    else:
      batch_graph_features = None

    # create new edge index
    num_nodes_list = [graph.num_nodes for graph in molecule_graphs]
    batch_edge_index = np.hstack(
      [graph.edge_index + prev_num_node for prev_num_node, graph \
        in zip([0] + num_nodes_list[:-1], molecule_graphs)]
    ).astype(int)

    # graph_index indicates which nodes belong to which graph
    graph_index = []
    for i, num_nodes in enumerate(num_nodes_list):
      graph_index.extend([i] * num_nodes)
    self.graph_index = np.array(graph_index, dtype=int)

    super().__init__(
        node_features=batch_node_features,
        edge_index=batch_edge_index,
        targets=batch_targets,
        edge_features=batch_edge_features,
        graph_features=batch_graph_features,
    )

    @staticmethod  # type: ignore
    def to_pyg_data(molecule_graphs: Sequence[MoleculeGraphData]):
      """Convert to PyTorch Geometric Batch instance

      Parameters
      ----------
      molecule_graphs : Sequence[MoleculeGraphData]
        List of MoleculeGraphData

      Returns
      -------
      torch_geometric.data.Batch
        Batch data of molecule graph for PyTorch Geometric
      """
      try:
        from torch_geometric.data import Batch
      except ModuleNotFoundError:
        raise ValueError(
            "This class requires PyTorch Geometric to be installed.")

      data_list = [mol_graph.to_pyg_data() for mol_graph in molecule_graphs]
      return Batch.from_data_list(data_list=data_list)
