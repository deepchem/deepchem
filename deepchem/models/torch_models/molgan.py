from typing import List, Tuple
# , Any, Optional,

# import torch
# from torch import nn
# from torch.nn import functional as F
# from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
from deepchem.models import WGAN
# from deepchem.models.layers import MolGANEncoderLayer


class BasicMolGANModel(WGAN):
    """
    Model for de-novo generation of small molecules based on work of Nicola De Cao et al. [1]_.
    It uses a GAN directly on graph data and a reinforcement learning objective to induce the network to generate molecules with certain chemical properties.
    Utilizes WGAN infrastructure; uses adjacency matrix and node features as inputs.
    Inputs need to be one-hot representation.

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 edges: int = 5,
                 vertices: int = 9,
                 nodes: int = 5,
                 embedding_dim: int = 10,
                 dropout_rate: float = 0.0,
                 **kwargs):
        """
        Initialize the model

        Parameters
        ----------
        edges: int, default 5
            Number of bond types includes BondType.Zero
        vertices: int, default 9
            Max number of atoms in adjacency and node features matrices
        nodes: int, default 5
            Number of atom types in node features matrix
        embedding_dim: int, default 10
            Size of noise input array
        dropout_rate: float, default = 0.
            Rate of dropout used across whole model
        name: str, default ''
            Name of the model
        """

        self.edges = edges
        self.vertices = vertices
        self.nodes = nodes
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        super(BasicMolGANModel, self).__init__(**kwargs)

    def get_noise_input_shape(self) -> Tuple[int]:
        """
        Return shape of the noise input used in generator

        Returns
        -------
        Tuple
            Shape of the noise input
        """

        return (self.embedding_dim,)

    def get_data_input_shapes(self) -> List:
        """
        Return input shape of the discriminator

        Returns
        -------
        List
            List of shapes used as an input for distriminator.
        """
        return [
            (self.vertices, self.vertices, self.edges),
            (self.vertices, self.nodes),
        ]
