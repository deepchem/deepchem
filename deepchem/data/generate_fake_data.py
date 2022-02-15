"""
Generates a Fake Graph Dataset
"""
import random
import deepchem as dc
import numpy as np

class FakeGraphDataset():
  """
  A fake dataset that randomly generates a collection of grapha. 
  The generated  dataset is of type NumpyDataset and contains graphs as 
  deepchem.feat.GraphData objects.

  This class is based on and it is almost similar to on torch_geometric.nn.datasets.FakeDataset.
  
  Parameters
  ----------
  num_graphs: int, default 1
    The number of graphs
  avg_num_nodes: int, default 10
    Average number of nodes in a graph
  avg_degree: int, default 4
    Average degree per node
  num_node_features: int, default 4
    Number of node features
  num_edge_features: int, default 3
    Number of edge features
  num_classes: int, default 2
    Number of classes in the dataset
  task: str, default 'graph'
    Whether to return graph-level or node-level labels
  is_undirected: bool, default True
    Whether the graph is undirected or not
  num_global_features: int, default 0
    A value greater than 0 requires  

  """
  def __init__(self, num_graphs=10, avg_num_nodes=10, avg_degree=4, num_node_features=4, num_edge_features=3, num_classes=2, task='graph', is_undirected=True, global_features=True):
    self.num_graphs = num_graphs
    self.avg_num_nodes = avg_num_nodes 
    self.avg_degree = avg_degree 
    self.num_node_features = num_node_features 
    self.num_edge_features = num_edge_features 
    self.num_classes = num_classes
    self.graph_dataset = self.generate_graphs()
    assert task in ['node', 'graph']
    self.task = task

  def generate_graph_dataset(self):
    """
    Returns a NumpyDataset
    """

  def generate_sample(self):
    """
    Generates a single Graph
    """
    num_nodes = int(np.random.normal(0, 3) + self.avg_num_nodes)

    # TODO for self.task = 'node'
    
    if self.task == 'graph':
      y = random.randint(0, self.num_classes-1)
    elif self.task == 'node':
      y = np.random.choice(self.num_classes, num_nodes)

    if self.task == 'graph':
      node_features = np.random.random(num_nodes, num_node_features) + (y * 2)
    
      edge_features = 
    


  
    data = Data()
  
    if self._num_classes > 0 and self.task == 'node':
        data.y = torch.randint(self._num_classes, (num_nodes, ))
    elif self._num_classes > 0 and self.task == 'graph':
        data.y = torch.tensor([random.randint(0, self._num_classes - 1)])
  
    data.edge_index = get_edge_index(num_nodes, num_nodes, self.avg_degree,
                                     self.is_undirected, remove_loops=True)
  
    if self.num_channels > 0 and self.task == 'graph':
        data.x = torch.randn(num_nodes, self.num_channels) + data.y
    elif self.num_channels > 0 and self.task == 'node':
        data.x = torch.randn(num_nodes,
                             self.num_channels) + data.y.unsqueeze(1)
    else:
        data.num_nodes = num_nodes
  
    if self.edge_dim > 1:
        if self.task == 'graph':
            data.edge_attr = torch.rand(data.num_edges,
                                        self.edge_dim) + data.y
        elif self.task == 'node':
            # no need to consider variance in edge distribution
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
    elif self.edge_dim == 1:

