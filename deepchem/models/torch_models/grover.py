import torch
import torch.nn as nn
from typing import List
from deepchem.models.torch_models.grover_layers import GroverTransEncoder


class GroverEmbedding(nn.Module):
    """GroverEmbedding layer.

    This layer is a simple wrapper over GroverTransEncoder layer for retrieving the embeddings from the GroverTransEncoder corresponding to the `embedding_output_type` chosen by the user.

    Parameters
    ----------
    edge_fdim: int
        the dimension of additional feature for edge/bond.
    node_fdim: int
        the dimension of additional feature for node/atom.
    depth: int
        Dynamic message passing depth for use in MPNEncoder
    undirected: bool
        The message passing is undirected or not
    num_mt_block: int
        the number of message passing blocks.
    num_head: int
        the number of attention heads.
    embedding_output_type: str
        the type of output aggregation after message passing.
                                            atom_messages:      True                      False
        - "none": no aggregation         output size:     (num_atoms, hidden_size)    (num_bonds, hidden_size)
        -"atom":  aggregating to atom  output size:     (num_atoms, hidden_size)    (num_atoms, hidden_size)
        -"bond": aggragating to bond.   output size:     (num_bonds, hidden_size)    (num_bonds, hidden_size)
        -"both": aggregating to atom&bond. output size:  (num_atoms, hidden_size)    (num_bonds, hidden_size)
                                                        (num_bonds, hidden_size)    (num_atoms, hidden_size)
    """

    def __init__(self,
                 node_fdim,
                 edge_fdim,
                 embedding_output_type,
                 hidden_size=128,
                 depth=1,
                 undirected=False,
                 dropout=0.2,
                 activation='relu',
                 num_mt_block=1,
                 num_heads=4,
                 bias=False,
                 res_connection=False):
        super(GroverEmbedding, self).__init__()
        self.embedding_output_type = embedding_output_type
        self.encoders = GroverTransEncoder(
            hidden_size=hidden_size,
            edge_fdim=edge_fdim,
            node_fdim=node_fdim,
            depth=depth,
            undirected=undirected,
            dropout=dropout,
            activation=activation,
            num_mt_block=num_mt_block,
            num_heads=num_heads,
            embedding_output_type=embedding_output_type,
            bias=bias,
            res_connection=res_connection)

    def forward(self, graph_batch: List[torch.Tensor]):
        """Forward function

        Parameters
        ----------
        graph_batch: List[torch.Tensor]
            A list containing f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        """
        output = self.encoders(graph_batch)
        if self.embedding_output_type == 'atom':
            return {
                "atom_from_atom": output[0],
                "atom_from_bond": output[1],
                "bond_from_atom": None,
                "bond_from_bond": None
            }  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            return {
                "atom_from_atom": None,
                "atom_from_bond": None,
                "bond_from_atom": output[0],
                "bond_from_bond": output[1]
            }  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            return {
                "atom_from_atom": output[0][0],
                "bond_from_atom": output[0][1],
                "atom_from_bond": output[1][0],
                "bond_from_bond": output[1][1]
            }
