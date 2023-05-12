from math import sqrt

import torch

from deepchem.feat.molecule_featurizers.conformer_featurizer import (
    full_atom_feature_dims,
    full_bond_feature_dims,
)


class AtomEncoder(torch.nn.Module):
    """
    Encodes atom features into embeddings based on the Open Graph Benchmark feature set in conformer_featurizer.

    Parameters
    ----------
    emb_dim : int
        The dimension that the returned embedding will have.
    padding : bool, optional (default=False)
        If true then the last index will be used for padding.

    Examples
    --------
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import full_atom_feature_dims
    >>> atom_encoder = AtomEncoder(emb_dim=32)
    >>> num_rows = 10
    >>> atom_features = torch.stack([
    ... torch.randint(low=0, high=dim, size=(num_rows,))
    ... for dim in full_atom_feature_dims
    ... ], dim=1)
    >>> atom_embeddings = atom_encoder(atom_features)
    """

    def __init__(self, emb_dim, padding=False):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for dim in full_atom_feature_dims:
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        """
        Reset the parameters of the atom embeddings.

        This method resets the weights of the atom embeddings by initializing
        them with a uniform distribution between -sqrt(3) and sqrt(3).
        """
        for embedder in self.atom_embedding_list:
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        """
        Compute the atom embeddings for the given atom features.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, num_atoms, num_features)
            The input atom features tensor.

        Returns
        -------
        x_embedding : torch.Tensor, shape (batch_size, num_atoms, emb_dim)
            The computed atom embeddings.
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                x_embedding += self.atom_embedding_list[i](x[:, i].long() + 1)
            else:
                x_embedding += self.atom_embedding_list[i](x[:, i].long())

        return x_embedding


class BondEncoder(torch.nn.Module):
    """
    Encodes bond features into embeddings based on the Open Graph Benchmark feature set in conformer_featurizer.

    Parameters
    ----------
    emb_dim : int
        The dimension that the returned embedding will have.
    padding : bool, optional (default=False)
        If true then the last index will be used for padding.

    Examples
    --------
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import full_bond_feature_dims
    >>> bond_encoder = BondEncoder(emb_dim=32)
    >>> num_rows = 10
    >>> bond_features = torch.stack([
    ... torch.randint(low=0, high=dim, size=(num_rows,))
    ... for dim in full_bond_feature_dims
    ... ], dim=1)
    >>> bond_embeddings = bond_encoder(bond_features)
    """

    def __init__(self, emb_dim, padding=False):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for dim in full_bond_feature_dims:
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        """
        Compute the bond embeddings for the given bond features.

        Parameters
        ----------
        edge_attr : torch.Tensor, shape (batch_size, num_edges, num_features)
            The input bond features tensor.

        Returns
        -------
        bond_embedding : torch.Tensor, shape (batch_size, num_edges, emb_dim)
            The computed bond embeddings.
        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if self.padding:
                bond_embedding += self.bond_embedding_list[i](
                    edge_attr[:, i].long() + 1)
            else:
                bond_embedding += self.bond_embedding_list[i](
                    edge_attr[:, i].long())

        return bond_embedding
