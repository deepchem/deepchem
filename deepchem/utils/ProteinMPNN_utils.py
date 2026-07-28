import numpy as np


def gather_edges(edges: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    """Gather edge features for each node's k-nearest neighbors.

    For every node, selects the edge feature vectors corresponding to its
    k-nearest neighbor indices from the full pairwise edge feature array.

    Parameters
    ----------
    edges : np.ndarray
        Full pairwise edge feature array of shape
        ``(batch, num_nodes, num_nodes, edge_features)``.
    neighbor_idx : np.ndarray
        k-nearest neighbor index array of shape
        ``(batch, num_nodes, k)``, where each entry is a node index.

    Returns
    -------
    np.ndarray
        Gathered edge features of shape
        ``(batch, num_nodes, k, edge_features)``.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187

    Examples
    --------
    >>> import numpy as np
    >>> edges = np.random.rand(2, 5, 5, 16)
    >>> neighbor_idx = np.random.randint(0, 5, (2, 5, 3))
    >>> out = gather_edges(edges, neighbor_idx)
    >>> out.shape
    (2, 5, 3, 16)
    """
    edge_features = edges.shape[-1]
    idx = np.expand_dims(neighbor_idx, axis=-1)
    idx = np.broadcast_to(idx, (*neighbor_idx.shape, edge_features))
    return np.take_along_axis(edges, idx, axis=2)


def gather_nodes(nodes: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    """Gather node features for each node's k-nearest neighbors.

    For every node, collects the feature vectors of its k-nearest neighbors
    by indexing into the node feature array with the provided neighbor indices.

    Parameters
    ----------
    nodes : np.ndarray
        Node feature array of shape ``(batch, num_nodes, node_features)``.
    neighbor_idx : np.ndarray
        k-nearest neighbor index array of shape
        ``(batch, num_nodes, k)``, where each entry is a node index.

    Returns
    -------
    np.ndarray
        Gathered neighbor node features of shape
        ``(batch, num_nodes, k, node_features)``.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187

    Examples
    --------
    >>> import numpy as np
    >>> nodes = np.random.rand(2, 5, 32)
    >>> neighbor_idx = np.random.randint(0, 5, (2, 5, 3))
    >>> out = gather_nodes(nodes, neighbor_idx)
    >>> out.shape
    (2, 5, 3, 32)
    """
    batch_size = neighbor_idx.shape[0]
    node_features = nodes.shape[2]
    neighbors_flat = neighbor_idx.reshape(batch_size, -1)
    idx = np.expand_dims(neighbors_flat, axis=-1)
    idx = np.broadcast_to(idx,
                          (batch_size, neighbors_flat.shape[1], node_features))
    neighbor_features = np.take_along_axis(nodes, idx, axis=1)
    return neighbor_features.reshape(*neighbor_idx.shape[:3], node_features)


def cat_neighbors_nodes(h_nodes: np.ndarray, h_neighbors: np.ndarray,
                        E_idx: np.ndarray) -> np.ndarray:
    """Concatenate neighboring node features with edge features.

    For each node and each of its k-nearest neighbors, gathers the neighbor's
    node feature vector and concatenates it with the corresponding edge feature
    vector. This combined representation is used as input to the message-passing
    layers in ProteinMPNN.

    Parameters
    ----------
    h_nodes : np.ndarray
        Node feature array of shape ``(batch, num_nodes, node_features)``.
    h_neighbors : np.ndarray
        Edge feature array for each node's k-nearest neighbors, of shape
        ``(batch, num_nodes, k, edge_features)``.
    E_idx : np.ndarray
        k-nearest neighbor index array of shape ``(batch, num_nodes, k)``,
        where each entry is a node index.

    Returns
    -------
    np.ndarray
        Concatenated features of shape
        ``(batch, num_nodes, k, edge_features + node_features)``.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187

    Examples
    --------
    >>> import numpy as np
    >>> h_nodes = np.random.rand(2, 5, 32)
    >>> h_neighbors = np.random.rand(2, 5, 3, 16)
    >>> E_idx = np.random.randint(0, 5, (2, 5, 3))
    >>> out = cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)
    >>> out.shape
    (2, 5, 3, 48)
    """
    h_nodes_gathered = gather_nodes(h_nodes, E_idx)
    return np.concatenate([h_neighbors, h_nodes_gathered], axis=-1)
