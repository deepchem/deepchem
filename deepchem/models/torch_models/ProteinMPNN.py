import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from deepchem.utils.ProteinMPNN_utils import gather_edges


class PositionalEncodings(nn.Module):
    """Relative positional encodings for residue pairs in the kNN graph.

    Encodes relative sequence offsets and chain membership between residue
    pairs into edge features used by the ProteinMPNN encoder.

    Parameters
    ----------
    num_embeddings : int
        Output dimension of the positional encoding.
    max_relative_feature : int, optional (default 32)
        Maximum absolute relative residue offset to encode.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187
    """

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute positional encodings for edge features.

        Parameters
        ----------
        offset : torch.Tensor
            Relative residue index offsets of shape
            ``(batch, num_nodes, k)``.
        mask : torch.Tensor
            Binary mask indicating same-chain residue pairs, of shape
            ``(batch, num_nodes, k)``.

        Returns
        -------
        torch.Tensor
            Positional edge embeddings of shape
            ``(batch, num_nodes, k, num_embeddings)``.
        """
        d = torch.clip(offset + self.max_relative_feature, 0,
                       2 * self.max_relative_feature) * mask + (1 - mask) * (
                           2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d.long(), 2 * self.max_relative_feature + 2)
        return self.linear(d_onehot.float())


class ProteinFeaturesLayer(nn.Module):
    """Extract k-nearest-neighbor graph edge features from backbone coordinates.

    This layer constructs a kNN graph from C-alpha distances and computes
    edge features from pairwise radial basis function (RBF) distances and
    relative positional encodings. These edge features are consumed by the
    ProteinMPNN encoder during message passing.

    Parameters
    ----------
    edge_features : int
        Output dimension of the edge feature embedding.
    num_positional_embeddings : int, optional (default 16)
        Dimension of relative positional encodings.
    num_rbf : int, optional (default 16)
        Number of radial basis functions used to encode distances.
    top_k : int, optional (default 30)
        Number of nearest neighbors per residue in the kNN graph.
    augment_eps : float, optional (default 0.0)
        Standard deviation of Gaussian noise added to coordinates during
        training. No augmentation is applied when set to 0.0.

    References
    ----------
    .. [1] Dauparas, J., et al. "Robust deep learning-based protein sequence
       design using ProteinMPNN." Science 378.6615 (2022): 49-56.
       https://doi.org/10.1126/science.add2187

    Examples
    --------
    >>> import torch
    >>> layer = ProteinFeaturesLayer(edge_features=128)
    >>> X = torch.randn(1, 10, 4, 3)
    >>> mask = torch.ones(1, 10)
    >>> residue_idx = torch.arange(10).unsqueeze(0)
    >>> chain_labels = torch.ones(1, 10)
    >>> E, E_idx = layer(X, mask, residue_idx, chain_labels)
    >>> E.shape[0]
    1
    """

    def __init__(self,
                 edge_features: int,
                 num_positional_embeddings: int = 16,
                 num_rbf: int = 16,
                 top_k: int = 30,
                 augment_eps: float = 0.0):
        super().__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def dist(self,
             X: torch.Tensor,
             mask: torch.Tensor,
             eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pairwise C-alpha distances and k-nearest neighbor indices.

        Parameters
        ----------
        X : torch.Tensor
            C-alpha coordinates of shape ``(batch, num_nodes, 3)``.
        mask : torch.Tensor
            Validity mask of shape ``(batch, num_nodes)``.
        eps : float, optional (default 1e-6)
            Small constant added for numerical stability in distance computation.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - D_neighbors: k-nearest neighbor distances of shape
              ``(batch, num_nodes, k)``.
            - E_idx: k-nearest neighbor indices of shape
              ``(batch, num_nodes, k)``.
        """
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, dim=3) + eps)
        D_max, _ = torch.max(D, dim=-1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
        return D_neighbors, E_idx

    def rbf(self, D: torch.Tensor) -> torch.Tensor:
        """Encode distances using radial basis functions.

        Parameters
        ----------
        D : torch.Tensor
            Pairwise distances of shape ``(batch, num_nodes, k)``.

        Returns
        -------
        torch.Tensor
            RBF-encoded distances of shape
            ``(batch, num_nodes, k, num_rbf)``.
        """
        D_min, D_max = 2.0, 22.0
        D_mu = torch.linspace(D_min, D_max, self.num_rbf, device=D.device)
        D_mu = D_mu.view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / self.num_rbf
        RBF = torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)
        return RBF

    def get_rbf(self, A: torch.Tensor, B: torch.Tensor,
                E_idx: torch.Tensor) -> torch.Tensor:
        """Compute RBF features for atom pairs at neighbor edges.

        Parameters
        ----------
        A : torch.Tensor
            Coordinates of the first atom type, shape ``(batch, num_nodes, 3)``.
        B : torch.Tensor
            Coordinates of the second atom type, shape ``(batch, num_nodes, 3)``.
        E_idx : torch.Tensor
            k-nearest neighbor index tensor of shape ``(batch, num_nodes, k)``.

        Returns
        -------
        torch.Tensor
            RBF-encoded pairwise distances of shape
            ``(batch, num_nodes, k, num_rbf)``.
        """
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :])**2, dim=-1) + 1e-6)
        D_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        return self.rbf(D_neighbors)

    def forward(
            self, X: torch.Tensor, mask: torch.Tensor,
            residue_idx: torch.Tensor,
            chain_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build kNN graph and compute edge features.

        Parameters
        ----------
        X : torch.Tensor
            Backbone coordinates of shape ``(batch, num_nodes, 4, 3)`` for
            atoms N, CA, C, and O.
        mask : torch.Tensor
            Validity mask of shape ``(batch, num_nodes)``.
        residue_idx : torch.Tensor
            Residue index tensor of shape ``(batch, num_nodes)``.
        chain_labels : torch.Tensor
            Chain identifier tensor of shape ``(batch, num_nodes)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - E: Edge feature tensor of shape
              ``(batch, num_nodes, k, edge_features)``.
            - E_idx: k-nearest neighbor index tensor of shape
              ``(batch, num_nodes, k)``.

        Examples
        --------
        >>> import torch
        >>> layer = ProteinFeaturesLayer(edge_features=128)
        >>> X = torch.randn(1, 10, 4, 3)
        >>> mask = torch.ones(1, 10)
        >>> residue_idx = torch.arange(10).unsqueeze(0)
        >>> chain_labels = torch.ones(1, 10)
        >>> E, E_idx = layer(X, mask, residue_idx, chain_labels)
        >>> len(E.shape)
        4
        """
        if self.augment_eps > 0 and self.training:
            X = X + self.augment_eps * torch.randn_like(X)

        N = X[:, :, 0, :]
        Ca = X[:, :, 1, :]
        C = X[:, :, 2, :]
        O_atom = X[:, :, 3, :]

        b = C - Ca
        c = N - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        D_neighbors, E_idx = self.dist(Ca, mask)

        atom_pairs = [(Ca, Ca), (N, N), (C, C), (O_atom, O_atom), (Cb, Cb),
                      (Ca, N), (Ca, C), (Ca, O_atom), (Ca, Cb), (N, C),
                      (N, O_atom), (N, Cb), (Cb, C), (Cb, O_atom), (O_atom, C),
                      (N, Ca), (C, Ca), (O_atom, Ca), (Cb, Ca), (C, N),
                      (O_atom, N), (Cb, N), (C, Cb), (O_atom, Cb), (C, O_atom)]
        RBF_all = [self.rbf(D_neighbors)]
        for A, B in atom_pairs[1:]:
            RBF_all.append(self.get_rbf(A, B, E_idx))
        RBF_all = torch.cat(RBF_all, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        d_chains = ((chain_labels[:, :, None] -
                     chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), dim=-1)
        E = self.norm_edges(self.edge_embedding(E))
        return E, E_idx
