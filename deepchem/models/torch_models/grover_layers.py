"""
Copied from https://github.com/tencent-ailab/grover/blob/0421d97a5e1bd1b59d1923e3afd556afbe4ff782/grover/model/layers.py
"""
from typing import List, Dict
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')
import numpy as np
from scipy import stats
from deepchem.models.torch_models.readout import GroverReadout
from deepchem.models.torch_models.layers import SublayerConnection, PositionwiseFeedForward


class GroverEmbedding(nn.Module):
    """GroverEmbedding layer.

    This layer is a simple wrapper over GroverTransEncoder layer for retrieving the embeddings from the GroverTransEncoder layer.

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
    """

    def __init__(self,
                 node_fdim,
                 edge_fdim,
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
        self.encoders = GroverTransEncoder(hidden_size=hidden_size,
                                           edge_fdim=edge_fdim,
                                           node_fdim=node_fdim,
                                           depth=depth,
                                           undirected=undirected,
                                           dropout=dropout,
                                           activation=activation,
                                           num_mt_block=num_mt_block,
                                           num_heads=num_heads,
                                           bias=bias,
                                           res_connection=res_connection)

    def forward(self, graph_batch: List[torch.Tensor]):
        """Forward function

        Parameters
        ----------
        graph_batch: List[torch.Tensor]
            A list containing f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        Returns
        -------
        embedding: Dict[str, torch.Tensor]
            Returns a dictionary of embeddings. The embeddings are:
         - atom_from_atom: node messages aggregated from node hidden states
         - bond_from_atom: bond messages aggregated from bond hidden states
         - atom_from_bond: node message aggregated from bond hidden states
         - bond_from_bond: bond messages aggregated from bond hidden states.
        """
        # TODO Explain in detail what the four outcompes are
        output = self.encoders(graph_batch)
        return {
            "atom_from_atom": output[0][0],
            "bond_from_atom": output[0][1],
            "atom_from_bond": output[1][0],
            "bond_from_bond": output[1][1]
        }


class GroverBondVocabPredictor(nn.Module):
    """Layer for learning contextual information for bonds.

    The layer is used in Grover architecture to learn contextual information of a bond by predicting
    the context of a bond from the bond embedding in a multi-class classification setting.
    The contextual information of a bond are encoded as strings (ex: '(DOUBLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)2').

    Example
    -------
    >>> from deepchem.models.torch_models.grover_layers import GroverBondVocabPredictor
    >>> num_bonds = 20
    >>> in_features, vocab_size = 16, 10
    >>> layer = GroverBondVocabPredictor(vocab_size, in_features)
    >>> embedding = torch.randn(num_bonds * 2, in_features)
    >>> result = layer(embedding)
    >>> result.shape
    torch.Size([20, 10])

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """

    def __init__(self, vocab_size: int, in_features: int = 128):
        """Initializes GroverBondVocabPredictor

        Parameters
        ----------
        vocab_size: int
            Size of vocabulary, used for number of classes in prediction.
        in_features: int, default: 128
            Input feature size of bond embeddings.
        """
        super(GroverBondVocabPredictor, self).__init__()
        self.linear = nn.Linear(in_features, vocab_size)
        self.linear_rev = nn.Linear(in_features, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        """
        Parameters
        ----------
        embeddings: torch.Tensor
            bond embeddings of shape (num_bond, in_features)

        Returns
        -------
        logits: torch.Tensor
            the prediction for each bond, (num_bond, vocab_size)
        """
        nm_bonds = embeddings.shape[0]
        # The bond and rev bond have even and odd ids respectively.
        ids1 = list(range(0, nm_bonds, 2))
        ids2 = list(range(1, nm_bonds, 2))
        logits = self.linear(embeddings[ids1]) + self.linear_rev(
            embeddings[ids2])
        return self.logsoftmax(logits)


class GroverAtomVocabPredictor(nn.Module):
    """Grover Atom Vocabulary Prediction Module.

    The GroverAtomVocabPredictor module is used for predicting atom-vocabulary
    for the self-supervision task in Grover architecture. In the self-supervision tasks,
    one task is to learn contextual-information of nodes (atoms).
    Contextual information are encoded as strings, like `C_N-DOUBLE1_O-SINGLE1`.
    The module accepts an atom encoding and learns to predict the contextual information
    of the atom as a multi-class classification problem.

    Example
    -------
    >>> from deepchem.models.torch_models.grover_layers import GroverAtomVocabPredictor
    >>> num_atoms, in_features, vocab_size = 30, 16, 10
    >>> layer = GroverAtomVocabPredictor(vocab_size, in_features)
    >>> embedding = torch.randn(num_atoms, in_features)
    >>> result = layer(embedding)
    >>> result.shape
    torch.Size([30, 10])

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """

    def __init__(self, vocab_size: int, in_features: int = 128):
        """Initializing Grover Atom Vocabulary Predictor

        Parameters
        ----------
        vocab_size: int
            size of vocabulary (vocabulary here is the total number of different possible contexts)
        in_features: int
            feature size of atom embeddings.
        """
        super(GroverAtomVocabPredictor, self).__init__()
        self.linear = nn.Linear(in_features, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        """
        Parameters
        ----------
        embeddings: torch.Tensor
            the atom embeddings of shape (vocab_size, in_features)

        Returns
        -------
        logits: torch.Tensor
            the prediction for each atom of shape (num_bond, vocab_size)
        """
        return self.logsoftmax(self.linear(embeddings))


class GroverFunctionalGroupPredictor(nn.Module):
    """The functional group prediction task for self-supervised learning.

    Molecules have functional groups in them. This module is used for predicting
    the functional group and the problem is formulated as an multi-label classification problem.

    Parameters
    ----------
    functional_group_size: int,
        size of functional group
    in_features: int,
        hidden_layer size, default 128

    Example
    -------
    >>> from deepchem.models.torch_models.grover_layers import GroverFunctionalGroupPredictor
    >>> in_features, functional_group_size = 8, 20
    >>> num_atoms, num_bonds = 10, 20
    >>> predictor = GroverFunctionalGroupPredictor(functional_group_size=20, in_features=8)
    >>> atom_scope, bond_scope = [(0, 3), (3, 3), (6, 4)], [(0, 5), (5, 4), (9, 11)]
    >>> embeddings = {}
    >>> embeddings['bond_from_atom'] = torch.randn(num_bonds, in_features)
    >>> embeddings['bond_from_bond'] = torch.randn(num_bonds, in_features)
    >>> embeddings['atom_from_atom'] = torch.randn(num_atoms, in_features)
    >>> embeddings['atom_from_bond'] = torch.randn(num_atoms, in_features)
    >>> result = predictor(embeddings, atom_scope, bond_scope)

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.

    """

    def __init__(self, functional_group_size: int, in_features=128):
        super(GroverFunctionalGroupPredictor, self).__init__()

        self.readout = GroverReadout(rtype="mean", in_features=in_features)
        self.linear_atom_from_atom = nn.Linear(in_features,
                                               functional_group_size)
        self.linear_atom_from_bond = nn.Linear(in_features,
                                               functional_group_size)
        self.linear_bond_from_atom = nn.Linear(in_features,
                                               functional_group_size)
        self.linear_bond_from_bond = nn.Linear(in_features,
                                               functional_group_size)

    def forward(self, embeddings: Dict, atom_scope: List, bond_scope: List):
        """
        The forward function for the GroverFunctionalGroupPredictor (semantic motif prediction) layer.
        It takes atom/bond embeddings produced from node and bond hidden states from GroverEmbedding module
        and the atom, bond scopes and produces prediction logits for different each embedding.
        The scopes are used to differentiate atoms/bonds belonging to a molecule in a batched molecular graph.

        Parameters
        ----------
        embedding: Dict
            The input embeddings organized as an dictionary. The input embeddings are output of GroverEmbedding layer.
        atom_scope: List
            The scope for atoms.
        bond_scope: List
            The scope for bonds

        Returns
        -------
        preds: Dict
            A dictionary containing the predicted logits of functional group from four different types of input embeddings. The key and their corresponding predictions
        are described below.
         - atom_from_atom - prediction logits from atom embeddings generated via node hidden states
         - atom_from_bond - prediction logits from atom embeddings generated via bond hidden states
         - bond_from_atom - prediction logits from bond embeddings generated via node hidden states
         - bond_from_bond - prediction logits from bond embeddings generated via bond hidden states
        """
        preds_bond_from_atom = self.linear_bond_from_atom(
            self.readout(embeddings["bond_from_atom"], bond_scope))
        preds_bond_from_bond = self.linear_bond_from_bond(
            self.readout(embeddings["bond_from_bond"], bond_scope))

        preds_atom_from_atom = self.linear_atom_from_atom(
            self.readout(embeddings["atom_from_atom"], atom_scope))
        preds_atom_from_bond = self.linear_atom_from_bond(
            self.readout(embeddings["atom_from_bond"], atom_scope))

        return {
            "atom_from_atom": preds_atom_from_atom,
            "atom_from_bond": preds_atom_from_bond,
            "bond_from_atom": preds_bond_from_atom,
            "bond_from_bond": preds_bond_from_bond
        }


def _index_select_nd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    Parameters
    ----------
    source: torch.Tensor
        A tensor of shape (num_bonds, hidden_size) containing message features.
    index: torch.Tensor
        A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond indices to select from source.

    Returns
    ----------
    message_features: torch.Tensor
        A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(
        -1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(
        final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def _select_neighbor_and_aggregate(feature, index):
    """The basic operation in message passing.

    Caution: the index_selec_ND would cause the reproducibility issue when performing the training on CUDA.

    The operation is like map and reduce. `index_select_nd` maps message features to bonds/atoms and
    this method aggregates the results by using `sum` as pooling operation. We can also add a configuration
    here to use `mean` as pooling operation but it is left to future implementation.

    References
    ----------
    See: https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    feature: np.array
        The candidate feature for aggregate. (n_nodes, hidden)
    index: np.array
        The selected index (neighbor indexes).

    Returns
    -------
    None
    """
    neighbor = _index_select_nd(feature, index)
    return neighbor.sum(dim=1)


class GroverMPNEncoder(nn.Module):
    """Performs Message Passing to generate encodings for the molecule.

    Parameters
    ----------
    atom_messages: bool
        True if encoding atom-messages else False.
    init_message_dim: int
        Dimension of embedding message.
    attach_feats: bool
        Set to `True` if additional features are passed along with node/edge embeddings.
    attached_feat_fdim: int
        Dimension of additional features when `attach_feats` is `True`
    undirected: bool
        If set to `True`, the graph is considered as an undirected graph.
    depth: int
        number of hops in a message passing iteration
    dynamic_depth: str, default: none
        If set to `uniform` for randomly sampling dynamic depth from an uniform distribution else if set to `truncnorm`, dynamic depth is sampled from a truncated normal distribution.
    input_layer: str
        If set to `fc`, adds an initial feed-forward layer. If set to `none`, does not add an initial feed forward layer.
    """

    # FIXME This layer is similar to DMPNNEncoderLayer and they
    # must be unified.
    def __init__(self,
                 atom_messages: bool,
                 init_message_dim: int,
                 hidden_size: int,
                 depth: int,
                 undirected: bool,
                 attach_feats: bool,
                 attached_feat_fdim: int = 0,
                 bias: bool = True,
                 dropout: float = 0.2,
                 activation: str = 'relu',
                 input_layer: str = 'fc',
                 dynamic_depth: str = 'none'):
        super(GroverMPNEncoder, self).__init__()
        if input_layer == 'none':
            assert init_message_dim == hidden_size
        assert dynamic_depth in [
            'uniform', 'truncnorm', 'none'
        ], 'dynamic depth should be one of uniform, truncnorm, none'
        self.init_message_dim = init_message_dim
        self.depth = depth
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.attached_feat = attach_feats

        assert dynamic_depth in [
            'none', 'truncnorm', 'uniform'
        ], 'If dynamic depth, it should be truncnorm or uniform'
        self.dynamic_depth = dynamic_depth

        self.dropout_layer = nn.Dropout(p=dropout)
        if activation == 'relu':
            self.act_func = nn.ReLU()
        else:
            raise ValueError('Only ReLU activation function is supported')

        if self.input_layer == "fc":
            self.W_i = nn.Linear(self.init_message_dim, hidden_size, bias=bias)
            w_h_input_size = hidden_size
        elif input_layer == 'none':
            w_h_input_size = self.init_message_dim

        if self.attached_feat:
            w_h_input_size += attached_feat_fdim

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, hidden_size, bias=bias)

    def forward(self,
                init_messages,
                init_attached_features,
                a2nei,
                a2attached,
                b2a=None,
                b2revb=None,
                adjs=None) -> torch.FloatTensor:
        if self.input_layer == 'fc':
            message = self.act_func(self.W_i(init_messages))
        elif self.input_layer == 'none':
            message = init_messages

        attached_feats = init_attached_features

        if self.training and self.dynamic_depth != 'none':
            if self.dynamic_depth == 'uniform':
                ndepth = abs(
                    int(np.random.uniform(self.depth - 3, self.depth + 3)))
            elif self.dynamic_depth == 'truncnorm':
                mu, sigma = self.depth, 1
                lower, upper = mu - 3 * sigma, mu + 3 * sigma
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma,
                                    loc=mu,
                                    scale=sigma)
                ndepth = int(X.rvs(1)[0])
        else:
            ndepth = self.depth

        for _ in range(ndepth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            nei_message = _select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_feat:
                attached_nei_feats = _select_neighbor_and_aggregate(
                    attached_feats, a2attached)
                a_message = torch.cat((nei_message, attached_nei_feats), dim=1)

            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_feat:
                    atom_rev_message = attached_feats[b2a[b2revb]]
                    rev_message = torch.cat((rev_message, atom_rev_message),
                                            dim=1)
                # Except reverse bond its-self(w) ! \sum_{k\in N(u) \ w}
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            else:
                message = a_message

            # FIXME When input_layer is none, for the first iteration of message passing, we should ideally
            # be using different weight matrix since message will be of shape num_bonds x f_bonds_dim
            # in the first iteration and in the subsequent iterations, it will be num_bonds x hidden_size
            message = self.W_h(message)

            message = self.dropout_layer(self.act_func(message))

        return message


class GroverAttentionHead(nn.Module):
    """Generates attention head using GroverMPNEncoder for generating query, key and value

    Parameters
    ----------
    hidden_size: int
        Dimension of hidden layer
    undirected: bool
        If set to `True`, the graph is considered as an undirected graph.
    depth: int
        number of hops in a message passing iteration
    atom_messages: bool
        True if encoding atom-messages else False.
    """

    def __init__(self,
                 hidden_size: int = 128,
                 bias: bool = True,
                 depth: int = 1,
                 dropout: float = 0.0,
                 undirected: bool = False,
                 atom_messages: bool = False):
        super(GroverAttentionHead, self).__init__()
        self.atom_messages = atom_messages

        # FIXME We assume that we are using a hidden layer to transform the initial atom message
        # and bond messages to hidden dimension size.
        self.mpn_q = GroverMPNEncoder(atom_messages=atom_messages,
                                      init_message_dim=hidden_size,
                                      attached_feat_fdim=hidden_size,
                                      hidden_size=hidden_size,
                                      bias=bias,
                                      depth=depth,
                                      dropout=dropout,
                                      undirected=undirected,
                                      attach_feats=False,
                                      input_layer='none',
                                      dynamic_depth='truncnorm')

        self.mpn_k = GroverMPNEncoder(atom_messages=atom_messages,
                                      init_message_dim=hidden_size,
                                      attached_feat_fdim=hidden_size,
                                      hidden_size=hidden_size,
                                      bias=bias,
                                      depth=depth,
                                      dropout=dropout,
                                      undirected=undirected,
                                      attach_feats=False,
                                      input_layer='none',
                                      dynamic_depth='truncnorm')

        self.mpn_v = GroverMPNEncoder(atom_messages=atom_messages,
                                      init_message_dim=hidden_size,
                                      attached_feat_fdim=hidden_size,
                                      hidden_size=hidden_size,
                                      bias=bias,
                                      depth=depth,
                                      dropout=dropout,
                                      undirected=undirected,
                                      attach_feats=False,
                                      input_layer='none',
                                      dynamic_depth='truncnorm')

    def forward(self, f_atoms, f_bonds, a2b, a2a, b2a, b2revb):
        if self.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
            b2a = b2a
            b2revb = b2revb
        else:
            # self.atom_messages is False
            init_messages = f_bonds
            init_attached_features = f_atoms
            a2nei = a2b
            a2attached = a2a
            b2a = b2a
            b2revb = b2revb

        q = self.mpn_q(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        k = self.mpn_k(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        v = self.mpn_v(init_messages=init_messages,
                       init_attached_features=init_attached_features,
                       a2nei=a2nei,
                       a2attached=a2attached,
                       b2a=b2a,
                       b2revb=b2revb)
        return q, k, v


class GroverMTBlock(nn.Module):
    """Message passing combined with transformer architecture

    The layer combines message passing performed using GroverMPNEncoder and uses it
    to generate query, key and value for multi-headed Attention block.

    Parameters
    ----------
    atom_messages: bool
        True if encoding atom-messages else False.
    input_dim: int
        Dimension of input features
    num_heads: int
        Number of attention heads
    depth: int
        Number of hops in a message passing iteration
    undirected: bool
        If set to `True`, the graph is considered as an undirected graph.
    """

    def __init__(self,
                 atom_messages: bool,
                 input_dim: int,
                 num_heads: int,
                 depth: int,
                 undirected: bool = False,
                 hidden_size: int = 128,
                 dropout: float = 0.0,
                 bias: bool = True,
                 res_connection: bool = True,
                 activation: str = 'relu'):
        super(GroverMTBlock, self).__init__()
        self.hidden_size = hidden_size
        self.atom_messages = atom_messages
        self.res_connection = res_connection
        self.num_heads = num_heads
        if activation == 'relu':
            self.act_func = nn.ReLU()
        else:
            raise ValueError('Only relu activation is supported')
        self.dropout_layer = nn.Dropout(p=dropout)

        # Note: Elementwise affine has to be consistent with the pre-training phase
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_size,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                bias=bias,
                                                batch_first=True)

        self.W_i = nn.Linear(input_dim, hidden_size, bias)
        self.W_o = nn.Linear(hidden_size * num_heads, hidden_size, bias)
        self.sublayer = SublayerConnection(hidden_size, dropout)
        self.attention_heads = nn.ModuleList()
        for i in range(num_heads):
            self.attention_heads.append(
                GroverAttentionHead(hidden_size=hidden_size,
                                    bias=bias,
                                    dropout=dropout,
                                    depth=depth,
                                    atom_messages=atom_messages,
                                    undirected=undirected))

    def forward(self, batch):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        if self.atom_messages:
            if f_atoms.shape[1] != self.hidden_size:
                f_atoms = self.W_i(f_atoms)
                f_atoms = self.dropout_layer(
                    self.layernorm(self.act_func(f_atoms)))

        else:
            if f_bonds.shape[1] != self.hidden_size:
                f_bonds = self.W_i(f_bonds)
                f_bonds = self.dropout_layer(
                    self.layernorm(self.act_func(f_bonds)))

        queries, keys, values = [], [], []
        for head in self.attention_heads:
            q, k, v = head(f_atoms,
                           f_bonds,
                           a2b=a2b,
                           b2a=b2a,
                           b2revb=b2revb,
                           a2a=a2a)
            queries.append(q.unsqueeze(1))
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)

        # multi-headed attention
        x_out, _ = self.attn(queries, keys, values)
        x_out = x_out.reshape(x_out.shape[0], -1)
        x_out = self.W_o(x_out)

        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                f_atoms = self.sublayer(f_atoms, x_out)
            else:
                f_bonds = self.sublayer(f_bonds, x_out)
        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        return batch


class GroverTransEncoder(nn.Module):
    """GroverTransEncoder for encoding a molecular graph

    The GroverTransEncoder layer is used for encoding a molecular graph.
    The layer returns 4 outputs. They are atom messages aggregated from atom hidden states,
    atom messages aggregated from bond hidden states, bond messages aggregated from atom hidden
    states, bond messages aggregated from bond hidden states.

    Parameters
    ----------
    hidden_size: int
        the hidden size of the model.
    edge_fdim: int
        the dimension of additional feature for edge/bond.
    node_fdim: int
        the dimension of additional feature for node/atom.
    depth: int
        Dynamic message passing depth for use in MPNEncoder
    undirected: bool
        The message passing is undirected or not
    dropout: float
        the dropout ratio
    activation: str
        the activation function
    num_mt_block: int
        the number of mt block.
    num_head: int
        the number of attention AttentionHead.
    bias: bool
        enable bias term in all linear layers.
    res_connection: bool
        enables the skip-connection in MTBlock.
    """

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 depth: int = 3,
                 undirected: bool = False,
                 num_mt_block: int = 2,
                 num_heads: int = 2,
                 hidden_size: int = 64,
                 dropout: float = 0.2,
                 res_connection: bool = True,
                 bias: bool = True,
                 activation: str = 'relu'):
        super(GroverTransEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.edge_fdim = edge_fdim
        self.node_fdim = node_fdim

        self.edge_blocks = nn.ModuleList()
        self.node_blocks = nn.ModuleList()

        for i in range(num_mt_block):
            if i == 0:
                node_input_fdim, edge_input_fdim = node_fdim, edge_fdim
            else:
                node_input_fdim, edge_input_fdim = hidden_size, hidden_size

            self.edge_blocks.append(
                GroverMTBlock(num_heads=num_heads,
                              input_dim=edge_input_fdim,
                              hidden_size=hidden_size,
                              activation=activation,
                              dropout=dropout,
                              bias=bias,
                              atom_messages=False,
                              res_connection=res_connection,
                              depth=depth,
                              undirected=undirected))
            self.node_blocks.append(
                GroverMTBlock(num_heads=num_heads,
                              input_dim=node_input_fdim,
                              hidden_size=hidden_size,
                              activation=activation,
                              dropout=dropout,
                              bias=bias,
                              atom_messages=True,
                              res_connection=res_connection,
                              depth=depth,
                              undirected=undirected))

        self.ffn_atom_from_atom = PositionwiseFeedForward(
            d_input=self.hidden_size + node_fdim,
            d_hidden=self.hidden_size * 4,
            d_output=self.hidden_size,
            n_layers=2,
            activation=activation,
            dropout_p=dropout)

        self.ffn_atom_from_bond = PositionwiseFeedForward(
            d_input=self.hidden_size + node_fdim,
            d_hidden=self.hidden_size * 4,
            d_output=self.hidden_size,
            n_layers=2,
            activation=activation,
            dropout_p=dropout)

        self.ffn_bond_from_atom = PositionwiseFeedForward(
            d_input=self.hidden_size + edge_fdim,
            d_hidden=self.hidden_size * 4,
            d_output=self.hidden_size,
            n_layers=2,
            activation=activation,
            dropout_p=dropout)

        self.ffn_bond_from_bond = PositionwiseFeedForward(
            d_input=self.hidden_size + edge_fdim,
            d_hidden=self.hidden_size * 4,
            d_output=self.hidden_size,
            n_layers=2,
            activation=activation,
            dropout_p=dropout)

        self.atom_from_atom_sublayer = SublayerConnection(size=self.hidden_size,
                                                          dropout_p=dropout)
        self.atom_from_bond_sublayer = SublayerConnection(size=self.hidden_size,
                                                          dropout_p=dropout)
        self.bond_from_atom_sublayer = SublayerConnection(size=self.hidden_size,
                                                          dropout_p=dropout)
        self.bond_from_bond_sublayer = SublayerConnection(size=self.hidden_size,
                                                          dropout_p=dropout)

        if activation == 'relu':
            self.act_func_node = nn.ReLU()
            self.act_func_edge = nn.ReLU()
        else:
            raise ValueError('Only relu activation is supported')

        self.dropout_layer = nn.Dropout(p=dropout)

    def _pointwise_feed_forward_to_atom_embedding(self, emb_output, atom_feat,
                                                  index, ffn_layer):
        aggr_output = _select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_feat, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_outputx

    def _pointwise_feed_forward_to_bond_embedding(self, emb_output, bond_feat,
                                                  a2nei, b2revb, ffn_layer):
        aggr_output = _select_neighbor_and_aggregate(emb_output, a2nei)
        aggr_output = self._remove_rev_bond_message(emb_output, aggr_output,
                                                    b2revb)
        aggr_outputx = torch.cat([bond_feat, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_outputx

    @staticmethod
    def _remove_rev_bond_message(original_message, aggr_message, b2revb):
        rev_message = original_message[b2revb]
        return aggr_message - rev_message

    def _atom_bond_transform(
            self,
            to_atom=True,  # False: to bond
            atomwise_input=None,
            bondwise_input=None,
            original_f_atoms=None,
            original_f_bonds=None,
            a2a=None,
            a2b=None,
            b2a=None,
            b2revb=None):
        """Transfer the output of atom/bond multi-head attention to the final atom/bond output.
        """

        if to_atom:
            # atom input to atom output
            atomwise_input, _ = self._pointwise_feed_forward_to_atom_embedding(
                atomwise_input, original_f_atoms, a2a, self.ffn_atom_from_atom)
            atom_in_atom_out = self.atom_from_atom_sublayer(
                None, atomwise_input)
            # bond to atom
            bondwise_input, _ = self._pointwise_feed_forward_to_atom_embedding(
                bondwise_input, original_f_atoms, a2b, self.ffn_atom_from_bond)
            bond_in_atom_out = self.atom_from_bond_sublayer(
                None, bondwise_input)
            return atom_in_atom_out, bond_in_atom_out
        else:  # to bond embeddings
            # atom input to bond output
            atom_list_for_bond = torch.cat([b2a.unsqueeze(dim=1), a2a[b2a]],
                                           dim=1)
            atomwise_input, _ = self._pointwise_feed_forward_to_bond_embedding(
                atomwise_input, original_f_bonds, atom_list_for_bond,
                b2a[b2revb], self.ffn_bond_from_atom)
            atom_in_bond_out = self.bond_from_atom_sublayer(
                None, atomwise_input)
            # bond input to bond output
            bond_list_for_bond = a2b[b2a]
            bondwise_input, _ = self._pointwise_feed_forward_to_bond_embedding(
                bondwise_input, original_f_bonds, bond_list_for_bond, b2revb,
                self.ffn_bond_from_bond)
            bond_in_bond_out = self.bond_from_bond_sublayer(
                None, bondwise_input)
            return atom_in_bond_out, bond_in_bond_out

    def forward(self, batch):
        """Forward layer

        Parameters
        ----------
        batch: Tuple
            A tuple of tensors representing grover attributes

        Returns
        -------
        embeddings: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            Embeddings for atom generated from hidden state of nodes and bonds and embeddings of bond generated from hidden states of nodes and bond.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

        node_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a
        edge_batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a

        original_f_atoms, original_f_bonds = f_atoms, f_bonds

        for nb in self.node_blocks:  # atom messages. Multi-headed attention
            node_batch = nb(node_batch)
        for eb in self.edge_blocks:  # bond messages. Multi-headed attention
            edge_batch = eb(edge_batch)

        atom_output, _, _, _, _, _, _, _ = node_batch  # atom hidden states
        _, bond_output, _, _, _, _, _, _ = edge_batch  # bond hidden states

        atom_embeddings = self._atom_bond_transform(
            to_atom=True,  # False: to bond
            atomwise_input=atom_output,
            bondwise_input=bond_output,
            original_f_atoms=original_f_atoms,
            original_f_bonds=original_f_bonds,
            a2a=a2a,
            a2b=a2b,
            b2a=b2a,
            b2revb=b2revb)

        bond_embeddings = self._atom_bond_transform(
            to_atom=False,  # False: to bond
            atomwise_input=atom_output,
            bondwise_input=bond_output,
            original_f_atoms=original_f_atoms,
            original_f_bonds=original_f_bonds,
            a2a=a2a,
            a2b=a2b,
            b2a=b2a,
            b2revb=b2revb)

        return ((atom_embeddings[0], bond_embeddings[0]), (atom_embeddings[1],
                                                           bond_embeddings[1]))
