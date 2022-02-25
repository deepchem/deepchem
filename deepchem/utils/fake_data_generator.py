"""
A fake data generator
"""
import random
import numpy as np
from deepchem.data import NumpyDataset
from deepchem.feat import GraphData


class FakeGraphGenerator:
  """Generates a random graphs which can be used for testing or other purposes.

  The generated graph supports both node-level and graph-level labels.

  Example
  -------
  >>> from deepchem.utils.fake_data_generator import FakeGraphGenerator
  >>> fgg  = FakeGraphGenerator(min_nodes=8, max_nodes=10,  n_node_features=5, avg_degree=8, n_edge_features=3, n_classes=2, task='graph', z=5)
  >>> graphs = fgg.sample(n_graphs=10)
  >>> type(graphs)
  <class 'deepchem.data.datasets.NumpyDataset'>
  >>> type(graphs.X[0])
  <class 'deepchem.feat.graph_data.GraphData'>
  >>> len(graphs) == 10  # num_graphs
  True

  Note
  ----
  The FakeGraphGenerator class is based on torch_geometric.dataset.FakeDataset
  class.
  """

  def __init__(self,
               min_nodes: int = 10,
               max_nodes: int = 10,
               n_node_features: int = 5,
               avg_degree: int = 4,
               n_edge_features: int = 3,
               n_classes: int = 2,
               task: str = 'graph',
               **kwargs):
    """
    Parameters
    ----------
    min_nodes: int, default 10
      Minimum number of permissible nodes in a graph
    max_nodes: int, default 10
      Maximum number of permissible nodes in a graph
    n_node_features: int, default 5
      Average number of node features in a graph
    avg_degree: int, default 4
      Average degree of the graph (avg_degree should be a positive number greater than the min_nodes)
    n_edge_features: int, default 3
      Average number of features in the edge
    task: str, default 'graph'
      Indicates node-level labels or graph-level labels
    kwargs: optional
      Additional graph attributes and their shapes , e.g. `global_features = 5`
    """
    assert avg_degree >= 1, "Average degree should be greater than 0"
    self.min_nodes = min_nodes
    self.max_nodes = max_nodes
    self.avg_degree = avg_degree
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_classes = n_classes
    self.task = task
    self.kwargs = kwargs

  def sample(self, n_graphs: int = 100) -> NumpyDataset:
    """Samples graphs

    Parameters
    ----------
    n_graphs: int, default 100
      Number of graphs to generate

    Returns
    -------
    graphs: NumpyDataset
      Generated Graphs
    """
    graphs, labels = [], []
    for i in range(n_graphs):
      n_nodes = random.randint(self.min_nodes, self.max_nodes)
      edge_index = generate_edge_index(n_nodes, self.avg_degree)
      n_edges = edge_index.shape[1]

      if self.task == 'graph':
        graph_label = random.randint(0, self.n_classes - 1)
        node_features = np.random.rand(n_nodes,
                                       self.n_node_features) + graph_label
        edge_features = np.random.rand(n_edges,
                                       self.n_edge_features) + graph_label
        kwargs = {}
        for feature_name, feature_shape in self.kwargs.items():
          kwargs[feature_name] = np.random.rand(1, feature_shape) + graph_label
        labels.append(graph_label)

      elif self.task == 'node':
        node_label = np.random.randint(0, self.n_classes - 1,
                                       n_nodes).astype(np.float64)
        node_features = np.random.rand(
            n_nodes, self.n_node_features) + node_label.reshape(-1, 1)
        # For a node-prediction task, label is not added to edge features and other global features
        # because label here is a node-level attribute and not a graph-level attribute
        edge_features = np.random.rand(n_edges, self.n_edge_features)
        kwargs = {}
        for feature_name, feature_shape in self.kwargs.items():
          kwargs[feature_name] = np.random.rand(1, feature_shape)
        kwargs['y'] = node_label

      graph = GraphData(node_features, edge_index, edge_features, **kwargs)
      graphs.append(graph)

      if self.task == 'graph':
        graph_dataset = NumpyDataset(X=np.array(graphs), y=np.array(labels))
      elif self.task == 'node':
        # In this case, the 'y' attribute of GraphData will contain the
        # node-level labels.
        graph_dataset = NumpyDataset(X=np.array(graphs))
    return graph_dataset


def generate_edge_index(n_nodes: int,
                        avg_degree: int,
                        remove_loops: bool = True) -> np.ndarray:
  """Returns source and destination nodes for `num_nodes * avg_degree` number of randomly
  generated edges. If remove_loops is True, then self-loops from the edge_index pairs
  are removed.

  Parameters
  ----------
  n_nodes: int
    Number of nodes in the graph
  avg_degree: int
    Average degree per node in a graph
  remove_loops: bool
    Remove self-loops in a graph
  """
  n_edges = n_nodes * avg_degree
  edge_index = np.random.randint(low=0, high=n_nodes, size=(2, n_edges))

  if remove_loops:
    edge_index = remove_self_loops(edge_index)
  return edge_index


def remove_self_loops(edge_index: np.ndarray) -> np.ndarray:
  """Removes self-loops from a given set of edges

  Parameters
  ----------
  edge_index: numpy.ndarray
    An numpy array of shape (2, |num_edges|) representing edges in a graph
  """
  mask = []
  for i in range(edge_index.shape[1]):
    if edge_index[0][i] != edge_index[1][i]:
      # not a self-loop
      mask.append(i)
  return edge_index[:, mask]
