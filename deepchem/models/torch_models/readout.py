from typing import List
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('The module requires PyTorch to be installed')

from deepchem.models.torch_models.attention import SelfAttention


class GroverReadout(nn.Module):
    """Performs readout on a batch of graph

    The readout module is used for performing readouts on batched graphs to
    convert node embeddings/edge embeddings into graph embeddings. It is used
    in the Grover architecture to generate a graph embedding from node and edge
    embeddings. The generate embedding can be used in downstream tasks like graph
    classification or graph prediction problems.

    Parameters
    ----------
    rtype: str
        Readout type, can be 'mean' or 'self-attention'
    in_features: int
        Size fof input features
    attn_hidden_size: int
        If readout type is attention, size of hidden layer in attention network.
    attn_out_size: int
        If readout type is attention, size of attention out layer.

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.readout import GroverReadout
    >>> n_nodes, n_features = 6, 32
    >>> readout = GroverReadout(rtype="mean")
    >>> embedding = torch.ones(n_nodes, n_features)
    >>> result = readout(embedding, scope=[(0, 6)])
    >>> result.size()
    torch.Size([1, 32])
    """

    def __init__(self,
                 rtype: str = 'mean',
                 in_features: int = 128,
                 attn_hidden_size: int = 32,
                 attn_out_size: int = 32):
        super(GroverReadout, self).__init__()
        self.cached_zero_vector = nn.Parameter(torch.zeros(in_features),
                                               requires_grad=False)
        self.rtype = rtype
        if rtype == "self_attention":
            self.attn = SelfAttention(hidden_size=attn_hidden_size,
                                      in_features=in_features,
                                      out_features=attn_out_size)

    def forward(self, graph_embeddings: torch.Tensor,
                scope: List[List]) -> torch.Tensor:
        """Given a batch node/edge embedding and a scope list, produce the graph-level embedding by scope.

        Parameters
        ----------
        embeddings: torch.Tensor
            The embedding matrix, num_nodes x in_features or num_edges x in_features.
        scope: List[List]
            A list, in which the element is a list [start, range]. `start` is the index,
            `range` is the length of scope. (start + range = end)

        Returns
        ----------
        graph_embeddings: torch.Tensor
            A stacked tensor containing graph embeddings of shape len(scope) x in_features if readout type is mean or len(scope) x attn_out_size when readout type is self-attention.
        """
        embeddings: List[torch.Tensor] = []
        for _, (a_start, a_size) in enumerate(scope):
            if a_size == 0:
                embeddings.append(self.cached_zero_vector)
            else:
                embedding = graph_embeddings.narrow(0, a_start, a_size)
                if self.rtype == "self_attention":
                    embedding, attn = self.attn(embedding)
                    embedding = embedding.flatten()
                elif self.rtype == "mean":
                    embedding = embedding.sum(dim=0) / a_size
                embeddings.append(embedding)

        graph_embeddings = torch.stack(embeddings, dim=0)
        return graph_embeddings
