import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Iterable, Tuple, Dict

from deepchem.models.losses import SparseSoftmaxCrossEntropy
from deepchem.models.torch_models import TorchModel
from deepchem.data import Dataset
from deepchem.utils.ProteinMPNN_utils import _gather_edges, _cat_neighbors_nodes, _gather_nodes

# Expected vocabulary size for standard ProteinMPNN (20 AAs + 1 Unknown)
NUM_AMINO_ACIDS: int = 21


class _PositionalEncodings(nn.Module):
    """Relative positional encodings for residue pairs in the kNN graph.

    Encodes the sequence offset between neighboring residues and whether they
    belong to the same chain into a learned edge feature vector.
    """

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        """Initialize positional encoding layer.

        Parameters
        ----------
        num_embeddings: int
            Output dimension of the positional encoding.
        max_relative_feature: int, default 32
            Maximum absolute sequence offset to encode. Offsets beyond this
            range are clipped before one-hot encoding.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute positional encodings for edge features.

        Parameters
        ----------
        offset: torch.Tensor
            Sequence index offsets between neighboring residues, with shape
            ``(batch, num_residues, k_neighbors)``.
        mask: torch.Tensor
            Binary mask indicating same-chain residue pairs, with shape
            ``(batch, num_residues, k_neighbors)``.

        Returns
        -------
        torch.Tensor
            Positional encoding vectors with shape
            ``(batch, num_residues, k_neighbors, num_embeddings)``.
        """
        d = torch.clip(offset + self.max_relative_feature, 0,
                       2 * self.max_relative_feature) * mask + (1 - mask) * (
                           2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d.long(), 2 * self.max_relative_feature + 2)
        return self.linear(d_onehot.float())


class _ProteinFeaturesLayer(nn.Module):
    """Extract k-nearest-neighbor graph edge features from backbone coordinates.

    Builds a kNN graph on C-alpha atoms and computes edge features from
    radial basis function (RBF) distance encodings between backbone atom
    pairs and relative positional encodings.
    """

    def __init__(self,
                 edge_features: int,
                 num_positional_embeddings: int = 16,
                 num_rbf: int = 16,
                 top_k: int = 30,
                 augment_eps: float = 0.0):
        """Initialize the protein feature extraction layer.

        Parameters
        ----------
        edge_features: int
            Output dimension of projected edge features.
        num_positional_embeddings: int, default 16
            Dimension of relative positional encodings.
        num_rbf: int, default 16
            Number of radial basis functions for distance encoding.
        top_k: int, default 30
            Number of nearest C-alpha neighbors per residue.
        augment_eps: float, default 0.0
            Standard deviation of Gaussian noise added to coordinates
            during training. Set to 0 to disable augmentation.
        """
        super().__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.embeddings = _PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self,
              X: torch.Tensor,
              mask: torch.Tensor,
              eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pairwise C-alpha distances and k-nearest neighbor indices.

        Parameters
        ----------
        X: torch.Tensor
            C-alpha coordinates with shape ``(batch, num_residues, 3)``.
        mask: torch.Tensor
            Residue validity mask with shape ``(batch, num_residues)``.
        eps: float, default 1e-6
            Small constant added for numerical stability in distance computation.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - D_neighbors: C-alpha distances to k nearest neighbors, shape
              ``(batch, num_residues, k_neighbors)``.
            - E_idx: Indices of k nearest neighbors, shape
              ``(batch, num_residues, k_neighbors)``.
        """
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, dim=3) + eps)
        D_max, _ = torch.max(D, dim=-1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        """Encode distances using radial basis functions.

        Parameters
        ----------
        D: torch.Tensor
            Inter-atomic distances with shape ``(batch, num_residues, k_neighbors)``.

        Returns
        -------
        torch.Tensor
            RBF-encoded distances with shape
            ``(batch, num_residues, k_neighbors, num_rbf)``.
        """
        D_min, D_max = 2.0, 22.0
        D_mu = torch.linspace(D_min, D_max, self.num_rbf, device=D.device)
        D_mu = D_mu.view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / self.num_rbf
        RBF = torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A: torch.Tensor, B: torch.Tensor,
                 E_idx: torch.Tensor) -> torch.Tensor:
        """Compute RBF features for atom pairs at neighbor edges.

        Parameters
        ----------
        A: torch.Tensor
            Coordinates of the first atom type with shape
            ``(batch, num_residues, 3)``.
        B: torch.Tensor
            Coordinates of the second atom type with shape
            ``(batch, num_residues, 3)``.
        E_idx: torch.Tensor
            kNN neighbor indices with shape
            ``(batch, num_residues, k_neighbors)``.

        Returns
        -------
        torch.Tensor
            RBF-encoded pairwise distances at neighbor edges with shape
            ``(batch, num_residues, k_neighbors, num_rbf)``.
        """
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :])**2, dim=-1) + 1e-6)
        D_neighbors = _gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(D_neighbors)

    def forward(
            self, X: torch.Tensor, mask: torch.Tensor,
            residue_idx: torch.Tensor,
            chain_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build kNN graph and compute edge features.

        Parameters
        ----------
        X: torch.Tensor
            Backbone atom coordinates with shape
            ``(batch, num_residues, 4, 3)`` for N, Ca, C, and O atoms.
        mask: torch.Tensor
            Residue validity mask with shape ``(batch, num_residues)``.
        residue_idx: torch.Tensor
            Sequential residue indices with shape ``(batch, num_residues)``.
        chain_labels: torch.Tensor
            Chain identifiers per residue with shape ``(batch, num_residues)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - E: Edge feature tensor with shape
              ``(batch, num_residues, k_neighbors, edge_features)``.
            - E_idx: kNN neighbor indices with shape
              ``(batch, num_residues, k_neighbors)``.
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

        D_neighbors, E_idx = self._dist(Ca, mask)

        atom_pairs = [(Ca, Ca), (N, N), (C, C), (O_atom, O_atom), (Cb, Cb),
                      (Ca, N), (Ca, C), (Ca, O_atom), (Ca, Cb), (N, C),
                      (N, O_atom), (N, Cb), (Cb, C), (Cb, O_atom), (O_atom, C),
                      (N, Ca), (C, Ca), (O_atom, Ca), (Cb, Ca), (C, N),
                      (O_atom, N), (Cb, N), (C, Cb), (O_atom, Cb), (C, O_atom)]
        RBF_all = [self._rbf(D_neighbors)]
        for A, B in atom_pairs[1:]:
            RBF_all.append(self._get_rbf(A, B, E_idx))
        RBF_all = torch.cat(RBF_all, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = _gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        d_chains = ((chain_labels[:, :, None] -
                     chain_labels[:, None, :]) == 0).long()
        E_chains = _gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), dim=-1)
        E = self.norm_edges(self.edge_embedding(E))
        return E, E_idx


class _PositionWiseFeedForward(nn.Module):
    """GELU feed-forward network used in encoder and decoder layers."""

    def __init__(self, num_hidden: int, num_ff: int):
        """Initialize the position-wise feed-forward network.

        Parameters
        ----------
        num_hidden: int
            Input and output hidden dimension.
        num_ff: int
            Intermediate feed-forward dimension.
        """
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward transformation.

        Parameters
        ----------
        h_V: torch.Tensor
            Node hidden states with shape ``(batch, num_residues, num_hidden)``.

        Returns
        -------
        torch.Tensor
            Transformed hidden states with the same shape as ``h_V``.
        """
        return self.W_out(self.act(self.W_in(h_V)))


class _EncoderLayer(nn.Module):
    """Encoder message-passing layer for ProteinMPNN.

    Updates both node and edge representations via two rounds of
    neighbor aggregation followed by feed-forward transformations.
    """

    def __init__(self,
                 num_hidden: int,
                 num_in: int,
                 dropout: float = 0.1,
                 scale: float = 30):
        """Initialize an encoder message-passing layer.

        Parameters
        ----------
        num_hidden: int
            Hidden dimension for node and edge representations.
        num_in: int
            Input feature dimension for the message-passing MLPs
            (concatenated node and edge features).
        dropout: float, default 0.1
            Dropout probability applied after residual connections.
        scale: float, default 30
            Normalization factor for aggregated neighbor messages.
        """
        super().__init__()
        self.num_hidden = num_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = _PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None,
        mask_attend: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one encoder message-passing step.

        Parameters
        ----------
        h_V: torch.Tensor
            Node hidden states with shape ``(batch, num_residues, num_hidden)``.
        h_E: torch.Tensor
            Edge hidden states with shape
            ``(batch, num_residues, k_neighbors, num_hidden)``.
        E_idx: torch.Tensor
            kNN neighbor indices with shape
            ``(batch, num_residues, k_neighbors)``.
        mask_V: torch.Tensor, optional
            Node validity mask with shape ``(batch, num_residues)``.
        mask_attend: torch.Tensor, optional
            Attention mask for neighbor aggregation with shape
            ``(batch, num_residues, k_neighbors)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated node hidden states ``h_V`` and edge hidden states ``h_E``.
        """
        h_EV = _cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, dim=-2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_EV = _cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class _DecoderLayer(nn.Module):
    """Decoder message-passing layer for autoregressive sequence prediction.

    Updates node representations using sequence-conditioned edge features
    with causal masking applied externally by the caller.
    """

    def __init__(self,
                 num_hidden: int,
                 num_in: int,
                 dropout: float = 0.1,
                 scale: float = 30):
        """Initialize a decoder message-passing layer.

        Parameters
        ----------
        num_hidden: int
            Hidden dimension for node representations.
        num_in: int
            Input feature dimension for the message-passing MLP
            (concatenated node and edge features).
        dropout: float, default 0.1
            Dropout probability applied after residual connections.
        scale: float, default 30
            Normalization factor for aggregated neighbor messages.
        """
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = _PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None,
        mask_attend: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform one decoder message-passing step.

        Parameters
        ----------
        h_V: torch.Tensor
            Node hidden states with shape ``(batch, num_residues, num_hidden)``.
        h_E: torch.Tensor
            Sequence-conditioned edge features with shape
            ``(batch, num_residues, k_neighbors, num_in)``.
        mask_V: torch.Tensor, optional
            Node validity mask with shape ``(batch, num_residues)``.
        mask_attend: torch.Tensor, optional
            Causal attention mask with shape
            ``(batch, num_residues, k_neighbors)``.

        Returns
        -------
        torch.Tensor
            Updated node hidden states with shape
            ``(batch, num_residues, num_hidden)``.
        """
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, dim=-2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V
        return h_V


class ProteinMPNN(nn.Module):
    """Protein Message Passing Neural Network for sequence design.

    Implements the encoder-decoder architecture from Dauparas et al. (2022)
    that predicts amino acid sequences from protein backbone coordinates.

    The model proceeds as follows:

    * Build a k-nearest-neighbor graph from backbone atom coordinates
    * Run encoder message-passing layers to update node and edge features
    * Autoregressively decode amino acid logits using a randomized decoding
      order over designable residues

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.ProteinMPNN import ProteinMPNN
    >>> model = ProteinMPNN(hidden_dim=128, num_encoder_layers=3,
    ...                   num_decoder_layers=3)
    >>> B, L = 1, 10
    >>> X = torch.randn(B, L, 4, 3)
    >>> S = torch.randint(0, 21, (B, L))
    >>> mask = torch.ones(B, L)
    >>> chain_M = torch.ones(B, L)
    >>> residue_idx = torch.arange(L).unsqueeze(0)
    >>> chain_encoding = torch.zeros(B, L, dtype=torch.long)
    >>> randn = torch.randn(B, L)
    >>> logits = model(X, S, mask, chain_M, residue_idx, chain_encoding, randn)
    >>> logits.shape
    torch.Size([1, 10, 21])

    References
    ----------
    .. [1] Dauparas, J. et al. Robust deep learning-based protein sequence
           design using ProteinMPNN. Science 378, 49-56 (2022).
           https://doi.org/10.1126/science.add2187
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 edge_features: int = 128,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 num_letters: int = NUM_AMINO_ACIDS,
                 vocab: int = NUM_AMINO_ACIDS,
                 k_neighbors: int = 30,
                 augment_eps: float = 0.05,
                 dropout: float = 0.1):
        """Initialize the ProteinMPNN model.

        Parameters
        ----------
        hidden_dim: int, default 128
            Hidden dimension for node and edge representations.
        edge_features: int, default 128
            Dimension of raw edge features before projection to hidden_dim.
        num_encoder_layers: int, default 3
            Number of encoder message-passing layers.
        num_decoder_layers: int, default 3
            Number of decoder message-passing layers.
        num_letters: int, default 21
            Number of output amino acid classes (20 standard + unknown).
        vocab: int, default 21
            Vocabulary size for the sequence embedding layer.
        k_neighbors: int, default 30
            Number of nearest C-alpha neighbors in the kNN graph.
        augment_eps: float, default 0.05
            Standard deviation of coordinate noise added during training.
        dropout: float, default 0.1
            Dropout probability in encoder and decoder layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.features = _ProteinFeaturesLayer(edge_features=edge_features,
                                              top_k=k_neighbors,
                                              augment_eps=augment_eps)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            _DecoderLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        use_input_decoding_order: bool = False,
        decoding_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing per-residue amino acid logits.

        Parameters
        ----------
        X: torch.Tensor
            Backbone atom coordinates with shape
            ``(batch, num_residues, 4, 3)`` for N, Ca, C, and O atoms.
        S: torch.Tensor
            Amino acid sequence indices with shape ``(batch, num_residues)``.
        mask: torch.Tensor
            Residue validity mask with shape ``(batch, num_residues)``.
        chain_M: torch.Tensor
            Design mask indicating which residues to predict, with shape
            ``(batch, num_residues)``. Non-zero values mark designable positions.
        residue_idx: torch.Tensor
            Sequential residue indices with shape ``(batch, num_residues)``.
        chain_encoding_all: torch.Tensor
            Chain identifiers per residue with shape ``(batch, num_residues)``.
        randn: torch.Tensor
            Random values used to determine autoregressive decoding order,
            with shape ``(batch, num_residues)``.
        use_input_decoding_order: bool, default False
            If True, use the provided ``decoding_order`` instead of
            generating one from ``randn``.
        decoding_order: torch.Tensor, optional
            Explicit decoding order with shape ``(batch, num_residues)``.
            Required when ``use_input_decoding_order`` is True.

        Returns
        -------
        torch.Tensor
            Per-residue amino acid logits with shape
            ``(batch, num_residues, num_letters)``.
        """
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros(E.shape[0],
                          E.shape[1],
                          self.hidden_dim,
                          device=device)
        h_E = self.W_e(E)

        mask_attend = _gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = _cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = _cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = _cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        if not use_input_decoding_order:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * torch.abs(randn))

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(decoding_order,
                                               num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij,biq,bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for layer in self.decoder_layers:
            h_ESV = _cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return logits


class ProteinMPNNModel(TorchModel):
    """DeepChem wrapper for ProteinMPNN sequence design.

    This class integrates the ``ProteinMPNN`` PyTorch model with DeepChem's
    training and evaluation infrastructure. It handles variable-length protein
    sequences via dynamic padding, similar to ``DMPNNModel``.

    The model expects featurized protein inputs as dictionaries containing
    backbone coordinates, sequence indices, masks, and chain metadata.
    Training uses ``SparseSoftmaxCrossEntropy`` loss over amino acid classes.

    Examples
    --------
    >>> import os
    >>> import deepchem as dc
    >>> from deepchem.feat.ProteinMPNN_featurizer import ProteinMPNNFeaturizer
    >>> # Featurize a PDB file
    >>> featurizer = ProteinMPNNFeaturizer()
    >>> features = featurizer.featurize(['path/to/protein.pdb'])
    >>> dataset = dc.data.NumpyDataset(X=features)
    >>> model = ProteinMPNNModel(batch_size=2)
    >>> loss = model.fit(dataset, nb_epoch=1)

    References
    ----------
    .. [1] Dauparas, J. et al. Robust deep learning-based protein sequence
           design using ProteinMPNN. Science 378, 49-56 (2022).
           https://doi.org/10.1126/science.add2187
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 edge_features: int = 128,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 k_neighbors: int = 30,
                 augment_eps: float = 0.05,
                 dropout: float = 0.1,
                 batch_size: int = 1,
                 **kwargs):
        """Initialize the ProteinMPNNModel.

        Parameters
        ----------
        hidden_dim: int, default 128
            Hidden dimension for node and edge representations.
        edge_features: int, default 128
            Dimension of raw edge features before projection.
        num_encoder_layers: int, default 3
            Number of encoder message-passing layers.
        num_decoder_layers: int, default 3
            Number of decoder message-passing layers.
        k_neighbors: int, default 30
            Number of nearest C-alpha neighbors in the graph.
        augment_eps: float, default 0.05
            Coordinate noise scale for training augmentation.
        dropout: float, default 0.1
            Dropout probability in encoder/decoder layers.
        batch_size: int, default 1
            Number of proteins per training batch.
        kwargs: dict
            Additional arguments passed to ``TorchModel``.
        """
        model = ProteinMPNN(
            hidden_dim=hidden_dim,
            edge_features=edge_features,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout,
        )
        loss = SparseSoftmaxCrossEntropy()
        super().__init__(model,
                         loss=loss,
                         output_types=['prediction', 'loss'],
                         batch_size=batch_size,
                         **kwargs)

    def _pad_batch(
            self,
            batch_dicts: List[Dict[str,
                                   np.ndarray]]) -> Tuple[torch.Tensor, ...]:
        """Pad a list of feature dictionaries to uniform sequence length.

        Parameters
        ----------
        batch_dicts : List[Dict[str, np.ndarray]]
            List of dictionaries extracted from the dataset, where each dict
            contains the mapped arrays for a single protein.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Batched tensors on the active device, in order:
            ``X, S, mask, chain_M, residue_idx, chain_encoding, randn``.

            - X: shape ``(batch, max_length, 4, 3)`` backbone coordinates
            - S: shape ``(batch, max_length)`` sequence indices
            - mask: shape ``(batch, max_length)`` residue validity mask
            - chain_M: shape ``(batch, max_length)`` design mask
            - residue_idx: shape ``(batch, max_length)`` residue indices
            - chain_encoding: shape ``(batch, max_length)`` chain labels
            - randn: shape ``(batch, max_length)`` decoding order noise
        """
        B = len(batch_dicts)
        L_max = max([d['X'].shape[0] for d in batch_dicts])

        X = np.zeros((B, L_max, 4, 3), dtype=np.float32)
        S = np.zeros((B, L_max), dtype=np.int64)
        mask = np.zeros((B, L_max), dtype=np.float32)
        chain_M = np.zeros((B, L_max), dtype=np.float32)
        residue_idx = np.zeros((B, L_max), dtype=np.int32)
        chain_encoding = np.zeros((B, L_max), dtype=np.int64)

        for i, d in enumerate(batch_dicts):
            L = d['X'].shape[0]
            X[i, :L] = d['X']
            S[i, :L] = d['S']
            mask[i, :L] = d['mask']
            chain_M[i, :L] = d['chain_M']
            residue_idx[i, :L] = d['residue_idx']
            chain_encoding[i, :L] = d['chain_encoding']

        randn = np.random.randn(B, L_max).astype(np.float32)

        device = self.device
        return (
            torch.from_numpy(X).float().to(device),
            torch.from_numpy(S).long().to(device),
            torch.from_numpy(mask).float().to(device),
            torch.from_numpy(chain_M).float().to(device),
            torch.from_numpy(residue_idx).long().to(device),
            torch.from_numpy(chain_encoding).long().to(device),
            torch.from_numpy(randn).float().to(device),
        )

    def _prepare_batch(
        self, batch: Tuple[List, List, List]
    ) -> Tuple[Tuple[torch.Tensor, ...], List[torch.Tensor],
               List[torch.Tensor]]:
        """Prepare batched model inputs from featurized protein data.

        Overrides the ``TorchModel._prepare_batch`` method to pad
        variable-length protein feature dictionaries into uniform tensors
        suitable for the ``ProteinMPNN`` forward pass.

        Parameters
        ----------
        batch: Tuple[List, List, List]
            Batch data from ``default_generator``, containing feature
            dictionaries, labels, and weights.

        Returns
        -------
        Tuple[Tuple[torch.Tensor, ...], List[torch.Tensor], List[torch.Tensor]]
            A tuple containing:
            - Model input tensors from ``_pad_batch``
            - Label tensors formatted for ``TorchModel``
            - Weight tensors formatted for ``TorchModel``
        """
        features_list, labels, weights = batch

        # Unpack if DeepChem nested the dictionaries inside an object array
        unpacked_features = []
        for x in features_list:
            if isinstance(x, (list, np.ndarray)) and len(x) > 0:
                unpacked_features.append(x[0])
            else:
                unpacked_features.append(x)

        inputs = self._pad_batch(unpacked_features)
        _, labels, weights = super(ProteinMPNNModel, self)._prepare_batch(
            ([], labels, weights))
        return inputs, labels, weights

    def default_generator(
        self,
        dataset: Dataset,
        epochs: int = 1,
        mode: str = 'fit',
        deterministic: bool = True,
        pad_batches: bool = False,
        **kwargs,
    ) -> Iterable[Tuple[List, List, List]]:
        """Create a generator that yields batches of protein features.

        Overrides the ``TorchModel.default_generator`` method to yield
        feature dictionaries from the dataset without additional conversion.
        Each feature dictionary should contain keys ``X``, ``S``, ``mask``,
        ``chain_M``, ``residue_idx``, and ``chain_encoding``.

        Parameters
        ----------
        dataset: Dataset
            DeepChem dataset containing feature dictionaries in X.
        epochs: int, default 1
            Number of passes over the dataset.
        mode: str, default 'fit'
            One of 'fit', 'predict', or 'uncertainty'.
        deterministic: bool, default True
            Whether to iterate in order or shuffle each epoch.
        pad_batches: bool, default False
            Whether to pad the final batch to ``batch_size``.
        kwargs: dict
            Additional keyword arguments passed to ``dataset.iterbatches``.

        Yields
        ------
        Tuple[List, List, List]
            ``([features], [labels], [weights])`` for each batch.
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                yield (list(X_b), [y_b], [w_b])
