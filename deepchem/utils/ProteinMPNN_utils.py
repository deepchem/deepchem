import torch


def _gather_edges(edges: torch.Tensor,
                  neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather edge features for each node's k-nearest neighbors.

    For every node, selects the edge feature vectors corresponding to its
    k-nearest neighbor indices from the full pairwise edge feature tensor.

    Parameters
    ----------
    edges : torch.Tensor
        Full pairwise edge feature tensor of shape
        ``(batch, num_nodes, num_nodes, edge_features)``.
    neighbor_idx : torch.Tensor
        k-nearest neighbor index tensor of shape
        ``(batch, num_nodes, k)``, where each entry is a node index.

    Returns
    -------
    torch.Tensor
        Gathered edge features of shape
        ``(batch, num_nodes, k, edge_features)``.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187

    Examples
    --------
    >>> import torch
    >>> edges = torch.rand(2, 5, 5, 16)
    >>> neighbor_idx = torch.randint(0, 5, (2, 5, 3))
    >>> out = _gather_edges(edges, neighbor_idx)
    >>> out.shape
    torch.Size([2, 5, 3, 16])
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def _gather_nodes(nodes: torch.Tensor,
                  neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather node features for each node's k-nearest neighbors.

    For every node, collects the feature vectors of its k-nearest neighbors
    by indexing into the node feature tensor with the provided neighbor indices.

    Parameters
    ----------
    nodes : torch.Tensor
        Node feature tensor of shape ``(batch, num_nodes, node_features)``.
    neighbor_idx : torch.Tensor
        k-nearest neighbor index tensor of shape
        ``(batch, num_nodes, k)``, where each entry is a node index.

    Returns
    -------
    torch.Tensor
        Gathered neighbor node features of shape
        ``(batch, num_nodes, k, node_features)``.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187

    Examples
    --------
    >>> import torch
    >>> nodes = torch.rand(2, 5, 32)
    >>> neighbor_idx = torch.randint(0, 5, (2, 5, 3))
    >>> out = _gather_nodes(nodes, neighbor_idx)
    >>> out.shape
    torch.Size([2, 5, 3, 32])
    """
    neighbors_flat = neighbor_idx.reshape(neighbor_idx.shape[0], -1)
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(*neighbor_idx.shape[:3], -1)


def _cat_neighbors_nodes(h_nodes: torch.Tensor, h_neighbors: torch.Tensor,
                         E_idx: torch.Tensor) -> torch.Tensor:
    """Concatenate neighboring node features with edge features.

    For each node and each of its k-nearest neighbors, gathers the neighbor's
    node feature vector and concatenates it with the corresponding edge feature
    vector. This combined representation is used as input to the message-passing
    layers in ProteinMPNN.

    Parameters
    ----------
    h_nodes : torch.Tensor
        Node feature tensor of shape ``(batch, num_nodes, node_features)``.
    h_neighbors : torch.Tensor
        Edge feature tensor for each node's k-nearest neighbors, of shape
        ``(batch, num_nodes, k, edge_features)``.
    E_idx : torch.Tensor
        k-nearest neighbor index tensor of shape ``(batch, num_nodes, k)``,
        where each entry is a node index.

    Returns
    -------
    torch.Tensor
        Concatenated features of shape
        ``(batch, num_nodes, k, edge_features + node_features)``.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187

    Examples
    --------
    >>> import torch
    >>> h_nodes = torch.rand(2, 5, 32)
    >>> h_neighbors = torch.rand(2, 5, 3, 16)
    >>> E_idx = torch.randint(0, 5, (2, 5, 3))
    >>> out = _cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)
    >>> out.shape
    torch.Size([2, 5, 3, 48])
    """
    h_nodes_gathered = _gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes_gathered], dim=-1)
