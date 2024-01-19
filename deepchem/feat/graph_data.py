from typing import Optional, Sequence

import numpy as np


class GraphData:
    """GraphData class

    This data class is almost same as `torch_geometric.data.Data
    <https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data>`_.

    Attributes
    ----------
    node_features: np.ndarray
        Node feature matrix with shape [num_nodes, num_node_features]
    edge_index: np.ndarray, dtype int
        Graph connectivity in COO format with shape [2, num_edges]
    edge_features: np.ndarray, optional (default None)
        Edge feature matrix with shape [num_edges, num_edge_features]
    node_pos_features: np.ndarray, optional (default None)
        Node position matrix with shape [num_nodes, num_dimensions].
    num_nodes: int
        The number of nodes in the graph
    num_node_features: int
        The number of features per node in the graph
    num_edges: int
        The number of edges in the graph
    num_edges_features: int, optional (default None)
        The number of features per edge in the graph

    Examples
    --------
    >>> import numpy as np
    >>> node_features = np.random.rand(5, 10)
    >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    >>> edge_features = np.random.rand(5, 5)
    >>> global_features = np.random.random(5)
    >>> graph = GraphData(node_features, edge_index, edge_features, z=global_features)
    >>> graph
    GraphData(node_features=[5, 10], edge_index=[2, 5], edge_features=[5, 5], z=[5])
    """

    def __init__(self,
                 node_features: np.ndarray,
                 edge_index: np.ndarray,
                 edge_features: Optional[np.ndarray] = None,
                 node_pos_features: Optional[np.ndarray] = None,
                 **kwargs):
        """
        Parameters
        ----------
        node_features: np.ndarray
            Node feature matrix with shape [num_nodes, num_node_features]
        edge_index: np.ndarray, dtype int
            Graph connectivity in COO format with shape [2, num_edges]
        edge_features: np.ndarray, optional (default None)
            Edge feature matrix with shape [num_edges, num_edge_features]
        node_pos_features: np.ndarray, optional (default None)
            Node position matrix with shape [num_nodes, num_dimensions].
        kwargs: optional
            Additional attributes and their values
        """
        # validate params
        if isinstance(node_features, np.ndarray) is False:
            raise ValueError('node_features must be np.ndarray.')

        if isinstance(edge_index, np.ndarray) is False:
            raise ValueError('edge_index must be np.ndarray.')
        elif issubclass(edge_index.dtype.type, np.integer) is False:
            raise ValueError('edge_index.dtype must contains integers.')
        elif edge_index.shape[0] != 2:
            raise ValueError('The shape of edge_index is [2, num_edges].')

        # np.max() method works only for a non-empty array, so size of the array should be non-zero
        elif (edge_index.size != 0) and (np.max(edge_index) >=
                                         len(node_features)):
            raise ValueError('edge_index contains the invalid node number.')

        if edge_features is not None:
            if isinstance(edge_features, np.ndarray) is False:
                raise ValueError('edge_features must be np.ndarray or None.')
            elif edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    'The first dimension of edge_features must be the same as the second dimension of edge_index.'
                )

        if node_pos_features is not None:
            if isinstance(node_pos_features, np.ndarray) is False:
                raise ValueError(
                    'node_pos_features must be np.ndarray or None.')
            elif node_pos_features.shape[0] != node_features.shape[0]:
                raise ValueError(
                    'The length of node_pos_features must be the same as the length of node_features.'
                )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_pos_features = node_pos_features
        self.kwargs = kwargs
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = edge_index.shape[1]
        if self.edge_features is not None:
            self.num_edge_features = self.edge_features.shape[1]

        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Returns a string containing the printable representation of the object"""
        cls = self.__class__.__name__
        node_features_str = str(list(self.node_features.shape))
        edge_index_str = str(list(self.edge_index.shape))
        if self.edge_features is not None:
            edge_features_str = str(list(self.edge_features.shape))
        else:
            edge_features_str = "None"

        out = "%s(node_features=%s, edge_index=%s, edge_features=%s" % (
            cls, node_features_str, edge_index_str, edge_features_str)
        # Adding shapes of kwargs
        for key, value in self.kwargs.items():
            if isinstance(value, np.ndarray):
                out += (', ' + key + '=' + str(list(value.shape)))
            elif isinstance(value, str):
                out += (', ' + key + '=' + value)
            elif isinstance(value, int) or isinstance(value, float):
                out += (', ' + key + '=' + str(value))
        out += ')'
        return out

    def to_pyg_graph(self):
        """Convert to PyTorch Geometric graph data instance

        Returns
        -------
        torch_geometric.data.Data
            Graph data for PyTorch Geometric

        Note
        ----
        This method requires PyTorch Geometric to be installed.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ModuleNotFoundError:
            raise ImportError(
                "This function requires PyTorch Geometric to be installed.")

        edge_features = self.edge_features
        if edge_features is not None:
            edge_features = torch.from_numpy(self.edge_features).float()
        node_pos_features = self.node_pos_features
        if node_pos_features is not None:
            node_pos_features = torch.from_numpy(self.node_pos_features).float()
        kwargs = {}
        for key, value in self.kwargs.items():
            kwargs[key] = torch.from_numpy(value).float()
        return Data(x=torch.from_numpy(self.node_features).float(),
                    edge_index=torch.from_numpy(self.edge_index).long(),
                    edge_attr=edge_features,
                    pos=node_pos_features,
                    **kwargs)

    def to_dgl_graph(self, self_loop: bool = False):
        """Convert to DGL graph data instance

        Returns
        -------
        dgl.DGLGraph
            Graph data for DGL
        self_loop: bool
            Whether to add self loops for the nodes, i.e. edges from nodes
            to themselves. Default to False.

        Note
        ----
        This method requires DGL to be installed.
        """
        try:
            import dgl
            import torch
        except ModuleNotFoundError:
            raise ImportError("This function requires DGL to be installed.")

        src = self.edge_index[0]
        dst = self.edge_index[1]

        g = dgl.graph(
            (torch.from_numpy(src).long(), torch.from_numpy(dst).long()),
            num_nodes=self.num_nodes)
        g.ndata['x'] = torch.from_numpy(self.node_features).float()

        if self.node_pos_features is not None:
            g.ndata['pos'] = torch.from_numpy(self.node_pos_features).float()
            g.edata['d'] = torch.norm(g.ndata['pos'][g.edges()[0]] -
                                      g.ndata['pos'][g.edges()[1]],
                                      p=2,
                                      dim=-1).unsqueeze(-1).detach()
        if self.edge_features is not None:
            g.edata['edge_attr'] = torch.from_numpy(self.edge_features).float()

        if self_loop:
            # This assumes that the edge features for self loops are full-zero tensors
            # In the future we may want to support featurization for self loops
            g.add_edges(np.arange(self.num_nodes), np.arange(self.num_nodes))

        return g

    def numpy_to_torch(self, device: str = 'cpu'):
        """Convert numpy arrays to torch tensors. This may be useful when you are using PyTorch Geometric with GraphData objects.

        Parameters
        ----------
        device : str
            Device to store the tensors. Default to 'cpu'.

        Example
        -------
        >>> num_nodes, num_node_features = 5, 32
        >>> num_edges, num_edge_features = 6, 32
        >>> node_features = np.random.random_sample((num_nodes, num_node_features))
        >>> edge_features = np.random.random_sample((num_edges, num_edge_features))
        >>> edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        >>> graph_data = GraphData(node_features, edge_index, edge_features)
        >>> graph_data = graph_data.numpy_to_torch()
        >>> print(type(graph_data.node_features))
        <class 'torch.Tensor'>
        """
        import copy

        import torch
        graph_copy = copy.deepcopy(self)

        graph_copy.node_features = torch.from_numpy(  # type: ignore
            self.node_features).float().to(device)
        graph_copy.edge_index = torch.from_numpy(  # type: ignore
            self.edge_index).long().to(device)
        if self.edge_features is not None:
            graph_copy.edge_features = torch.from_numpy(  # type: ignore
                self.edge_features).float().to(device)
        else:
            graph_copy.edge_features = None
        if self.node_pos_features is not None:
            graph_copy.node_pos_features = torch.from_numpy(  # type: ignore
                self.node_pos_features).float().to(device)
        else:
            graph_copy.node_pos_features = None

        graph_copy.kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).to(device)
                graph_copy.kwargs[key] = value
                setattr(graph_copy, key, value)

        return graph_copy

    def subgraph(self, nodes):
        """Returns a subgraph of `nodes` indicies.

        Parameters
        ----------
        nodes : list, iterable
            A list of node indices to be included in the subgraph.

        Returns
        -------
        subgraph_data : GraphData
            A new GraphData object containing the subgraph induced on `nodes`.

        Example
        -------
        >>> import numpy as np
        >>> from deepchem.feat.graph_data import GraphData
        >>> node_features = np.random.rand(5, 10)
        >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
        >>> edge_features = np.random.rand(5, 3)
        >>> graph_data = GraphData(node_features, edge_index, edge_features)
        >>> nodes = [0, 2, 4]
        >>> subgraph_data, node_mapping = graph_data.subgraph(nodes)
        """
        nodes = set(nodes)
        if not nodes.issubset(range(self.num_nodes)):
            raise ValueError("Some nodes are not in the original graph")

        # Create a mapping from the original node indices to the new node indices
        node_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(nodes)
        }

        # Filter and reindex node features
        subgraph_node_features = self.node_features[list(nodes)]

        # Filter and reindex edge indices and edge features
        subgraph_edge_indices = []
        subgraph_edge_features = []
        if self.edge_features is not None:
            for i in range(self.num_edges):
                src, dest = self.edge_index[:, i]
                if src in nodes and dest in nodes:
                    subgraph_edge_indices.append(
                        (node_mapping[src], node_mapping[dest]))
                    subgraph_edge_features.append(self.edge_features[i])

        subgraph_edge_index = np.array(subgraph_edge_indices, dtype=np.int64).T
        subgraph_edge_features = np.array(subgraph_edge_features)

        subgraph_data = GraphData(node_features=subgraph_node_features,
                                  edge_index=subgraph_edge_index,
                                  edge_features=subgraph_edge_features,
                                  **self.kwargs)

        return subgraph_data, node_mapping


class BatchGraphData(GraphData):
    """Batch GraphData class

    Attributes
    ----------
    node_features: np.ndarray
        Concatenated node feature matrix with shape [num_nodes, num_node_features].
        `num_nodes` is total number of nodes in the batch graph.
    edge_index: np.ndarray, dtype int
        Concatenated graph connectivity in COO format with shape [2, num_edges].
        `num_edges` is total number of edges in the batch graph.
    edge_features: np.ndarray, optional (default None)
        Concatenated edge feature matrix with shape [num_edges, num_edge_features].
        `num_edges` is total number of edges in the batch graph.
    node_pos_features: np.ndarray, optional (default None)
        Concatenated node position matrix with shape [num_nodes, num_dimensions].
        `num_nodes` is total number of edges in the batch graph.
    num_nodes: int
        The number of nodes in the batch graph.
    num_node_features: int
        The number of features per node in the graph.
    num_edges: int
        The number of edges in the batch graph.
    num_edges_features: int, optional (default None)
        The number of features per edge in the graph.
    graph_index: np.ndarray, dtype int
        This vector indicates which graph the node belongs with shape [num_nodes,].

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.feat.graph_data import GraphData
    >>> node_features_list = np.random.rand(2, 5, 10)
    >>> edge_index_list = np.array([
    ...    [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
    ...    [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
    ... ], dtype=int)
    >>> user_defined_attribute = np.array([0, 1])
    >>> graph_list = [GraphData(node_features, edge_index, attribute=user_defined_attribute)
    ...     for node_features, edge_index in zip(node_features_list, edge_index_list)]
    >>> batch_graph = BatchGraphData(graph_list=graph_list)
    """

    def __init__(self, graph_list: Sequence[GraphData]):
        """
        Parameters
        ----------
        graph_list: Sequence[GraphData]
            List of GraphData
        """
        # stack features
        batch_node_features = np.vstack(
            [graph.node_features for graph in graph_list])

        # before stacking edge_features or node_pos_features,
        # we should check whether these are None or not
        if graph_list[0].edge_features is not None:
            batch_edge_features: Optional[np.ndarray] = np.vstack(
                [graph.edge_features for graph in graph_list])  # type: ignore
        else:
            batch_edge_features = None

        if graph_list[0].node_pos_features is not None:
            batch_node_pos_features: Optional[np.ndarray] = np.vstack([
                graph.node_pos_features for graph in graph_list  # type: ignore
            ])

        else:
            batch_node_pos_features = None

        # create new edge index
        # number of nodes in each graph
        num_nodes_list = [graph.num_nodes for graph in graph_list]
        # cumulative number of nodes for each graph. This is necessary because the values in edge_index are node indices of all of the graphs in graph_list and so we need to offset the indices by the number of nodes in the previous graphs.
        cum_num_nodes_list = np.cumsum([0] + num_nodes_list)[:-1]
        # columns are the edge index, values are the node index
        batch_edge_index = np.hstack([
            graph.edge_index + cum_num_nodes
            for cum_num_nodes, graph in zip(cum_num_nodes_list, graph_list)
        ])

        # graph_index indicates which nodes belong to which graph
        graph_index = []
        for i, num_nodes in enumerate(num_nodes_list):
            graph_index.extend([i] * num_nodes)
        self.graph_index = np.array(graph_index)

        # Batch user defined attributes
        kwargs = {}
        user_defined_attribute_names = self._get_user_defined_attributes(
            graph_list[0])
        for name in user_defined_attribute_names:
            kwargs[name] = np.vstack(
                [getattr(graph, name) for graph in graph_list])

        super().__init__(node_features=batch_node_features,
                         edge_index=batch_edge_index,
                         edge_features=batch_edge_features,
                         node_pos_features=batch_node_pos_features,
                         **kwargs)

    def _get_user_defined_attributes(self, graph_data: GraphData):
        """A GraphData object can have user defined attributes but the attribute name of those
        are unknown since it can be arbitary. This method helps to find user defined attribute's
        name by making a list of known graph data attributes and finding other user defined
        attributes via `vars` method. The user defined attributes are attributes other than
        `node_features`, `edge_index`, `edge_features`, `node_pos_features`, `kwargs`, `num_nodes`,
        `num_node_features`, `num_edges`, `num_edge_features` as these are graph data attributes."""
        graph_data_attributes = [
            'node_features', 'edge_index', 'edge_features', 'node_pos_features',
            'kwargs', 'num_nodes', 'num_node_features', 'num_edges',
            'num_edge_features'
        ]
        user_defined_attribute_names = []
        for arg in vars(graph_data):
            if arg not in graph_data_attributes:
                user_defined_attribute_names.append(arg)
        return user_defined_attribute_names

    def numpy_to_torch(self, device: str = "cpu"):
        """
        Convert numpy arrays to torch tensors for BatchGraphData. BatchGraphData is very similar to GraphData, but it combines all graphs into a single graph object and it has an additional attribute `graph_index` which indicates which nodes belong to which graph.

        Parameters
        ----------
        device : str
            Device to store the tensors. Default to 'cpu'.

        Example
        -------
        >>> num_nodes, num_node_features = 5, 32
        >>> num_edges, num_edge_features = 6, 32
        >>> node_features = np.random.random_sample((num_nodes, num_node_features))
        >>> edge_features = np.random.random_sample((num_edges, num_edge_features))
        >>> edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        >>> graph_data = GraphData(node_features, edge_index, edge_features)
        >>> node_features2 = np.random.random_sample((num_nodes, num_node_features))
        >>> edge_features2 = np.random.random_sample((num_edges, num_edge_features))
        >>> edge_index2 = np.random.randint(0, num_nodes, (2, num_edges))
        >>> graph_data2 = GraphData(node_features2, edge_index2, edge_features2)
        >>> batch_graph_data = BatchGraphData([graph_data, graph_data2])
        >>> batch_graph_data = batch_graph_data.numpy_to_torch()
        >>> print(type(batch_graph_data.node_features))
        <class 'torch.Tensor'>
        """
        import torch
        graph_copy = super().numpy_to_torch(device)

        graph_index = torch.from_numpy(
            self.graph_index).long().to(  # type: ignore
                device)
        graph_copy.graph_index = graph_index

        return graph_copy


def shortest_path_length(graph_data, source, cutoff=None):
    """Compute the shortest path lengths from source to all reachable nodes in a GraphData object.

    This function only works with undirected graphs.

    Parameters
    ----------
    graph_data : GraphData
        GraphData object containing the graph information

    source : int
       Starting node index for path

    cutoff : int, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : dict
        Dict of node index and shortest path length from source to that node.

    Examples
    --------
    >>> import numpy as np
    >>> node_features = np.random.rand(5, 10)
    >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    >>> graph_data = GraphData(node_features, edge_index)
    >>> shortest_path_length(graph_data, 0)
    {0: 0, 1: 1, 2: 2, 3: 2, 4: 1}
    >>> shortest_path_length(graph_data, 0, cutoff=1)
    {0: 0, 1: 1, 4: 1}

    """
    if source >= graph_data.num_nodes:
        raise ValueError(f"Source {source} is not in graph_data")
    if cutoff is None:
        cutoff = float("inf")

    # Convert edge_index to adjacency list
    adj_list = [[] for _ in range(graph_data.num_nodes)]
    for i in range(graph_data.num_edges):
        src, dest = graph_data.edge_index[:, i]
        adj_list[src].append(dest)
        adj_list[dest].append(src)  # Assuming undirected graph

    # Breadth-first search
    visited = np.full(graph_data.num_nodes, False)
    distances = np.full(graph_data.num_nodes, np.inf)
    queue = [source]
    visited[source] = True
    distances[source] = 0

    while queue:
        node = queue.pop(0)
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                distances[neighbor] = distances[node] + 1
                if distances[neighbor] < cutoff:
                    queue.append(neighbor)

    return {i: int(d) for i, d in enumerate(distances) if d <= cutoff}
