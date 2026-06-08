import torch


def _gather_edges(edges: torch.Tensor,
                  neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather edge features at neighbor indices.

    Parameters
    ----------
    edges: torch.Tensor
        Edge features of shape ``(B, N, N, C)``.
    neighbor_idx: torch.Tensor
        Neighbor indices of shape ``(B, N, K)``.

    Returns
    -------
    torch.Tensor
        Neighbor edge features of shape ``(B, N, K, C)``.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.proteinMPNN import _gather_edges
    >>> edges = torch.randn(1, 4, 4, 8)
    >>> neighbor_idx = torch.tensor([[[1, 2, 3, 0], [0, 2, 3, 1],
    ...                                 [0, 1, 3, 2], [0, 1, 2, 3]]])
    >>> out = _gather_edges(edges, neighbor_idx)
    >>> out.shape
    torch.Size([1, 4, 4, 8])
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def _gather_nodes(nodes: torch.Tensor,
                  neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather node features at neighbor indices.

    Parameters
    ----------
    nodes: torch.Tensor
        Node features of shape ``(B, N, C)``.
    neighbor_idx: torch.Tensor
        Neighbor indices of shape ``(B, N, K)``.

    Returns
    -------
    torch.Tensor
        Neighbor node features of shape ``(B, N, K, C)``.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.proteinMPNN import _gather_nodes
    >>> nodes = torch.randn(1, 4, 16)
    >>> neighbor_idx = torch.tensor([[[1, 2, 3, 0], [0, 2, 3, 1],
    ...                                 [0, 1, 3, 2], [0, 1, 2, 3]]])
    >>> out = _gather_nodes(nodes, neighbor_idx)
    >>> out.shape
    torch.Size([1, 4, 4, 16])
    """
    neighbors_flat = neighbor_idx.reshape(neighbor_idx.shape[0], -1)
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(*neighbor_idx.shape[:3], -1)


def _cat_neighbors_nodes(h_nodes: torch.Tensor, h_neighbors: torch.Tensor,
                         E_idx: torch.Tensor) -> torch.Tensor:
    """Concatenate node features with neighbor edge features.

    Parameters
    ----------
    h_nodes: torch.Tensor
        Node hidden states of shape ``(B, N, C)``.
    h_neighbors: torch.Tensor
        Neighbor edge features of shape ``(B, N, K, C_e)``.
    E_idx: torch.Tensor
        Neighbor indices of shape ``(B, N, K)``.

    Returns
    -------
    torch.Tensor
        Concatenated features of shape ``(B, N, K, C + C_e)``.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.proteinMPNN import _cat_neighbors_nodes
    >>> h_nodes = torch.randn(1, 4, 16)
    >>> h_neighbors = torch.randn(1, 4, 4, 32)
    >>> E_idx = torch.tensor([[[1, 2, 3, 0], [0, 2, 3, 1],
    ...                        [0, 1, 3, 2], [0, 1, 2, 3]]])
    >>> out = _cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)
    >>> out.shape
    torch.Size([1, 4, 4, 48])
    """
    h_nodes_gathered = _gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes_gathered], dim=-1)
