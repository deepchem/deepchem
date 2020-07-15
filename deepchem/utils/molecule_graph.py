from typing import Optional
import numpy as np


class MoleculeGraphData(object):
    """Molecule Graph Data class for sparse pattern"""

    def __init__(self,
                 node_features: Optional[np.ndarray] = None,
                 edge_index: Optional[np.ndarray] = None,
                 edge_features: Optional[np.ndarray] = None,
                 graph_features: Optional[np.ndarray] = None,
                 targets : Optional[np.ndarray] = None,):
        """

        Parameters
        ----------
        node_features : np.ndarray, optional (default None)
          Node feature matrix with shape [num_nodes, num_node_features]
        edge_index : np.ndarray, optional (default None)
          Graph connectivity in COO format with shape [2, num_edges]
        edge_features : np.ndarray, optional (default None)
          Edge feature matrix with shape [num_edges, num_edge_features]
        graph_features : np.ndarray, optional (default None)
          Graph feature vector with shape [num_graph_features,]
        targets : np.ndarray, optional (default None)
          Graph or node targets with arbitrary shape
        """
        super(MoleculeGraphData, self).__init__()
        # validate params
        if node_features is not None and isinstance(node_features, np.ndarray) is False:
            raise ValueError('node_features must be np.ndarray or None.')
        if edge_index is not None:
            if isinstance(edge_index, np.ndarray) is False:
                raise ValueError('edge_index must be np.ndarray or None.')
            elif edge_index.shape[0] != 2:
                raise ValueError('The shape of edge_index is [2, num_edges].')
        if edge_features is not None:
            if instance(edge_features, np.ndarray) is False:
                raise ValueError('edge_features must be np.ndarray or None.')
            elif edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError('The first dimension of edge_features must be the \
                    same as the second dimension of edge_index.')
        if graph_features is not None and isinstance(graph_features, np.ndarray) is False:
            raise ValueError('graph_features must be np.ndarray or None.')
        if targets is not None and isinstance(targets, np.ndarray) is False:
            raise ValueError('y must be np.ndarray or None.')

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.graph_features = graph_features
        self.targets = targets
        self.num_nodes, self.num_node_features = None, None
        self.num_edges, self.num_edge_features = None, None
        if self.node_features is not None:
            self.num_nodes, self.num_node_features = self.node_features.shape
        if self.node_features is not None:
            self.num_edges, self.num_edge_features = self.edge_features.shape
