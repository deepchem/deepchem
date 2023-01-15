"""
Copied from https://github.com/tencent-ailab/grover/blob/0421d97a5e1bd1b59d1923e3afd556afbe4ff782/grover/model/layers.py
"""
from deepchem.models.torch_models.layers import SublayerConnection, PositionwiseFeedForward


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
    """Performs Message Passing to encodes a molecule

    """

    # FIXME This layer is similar to DMPNNEncoderLayer and they
    # must be unified.
    def __init__(self,
                 atom_messages: bool,
                 init_message_dim: int,
                 attached_feat_fdim: int,
                 hidden_size: int,
                 bias: bool,
                 depth: int,
                 undirected: bool,
                 attach_feat: bool,
                 dropout: float = 0.2,
                 activation: str = 'relu',
                 input_layer: str = 'fc',
                 dynamic_depth: str = 'none'):
        super(GroverMPNEncoder, self).__init__()
        if input_layer == 'none':
            assert init_message_dim == hidden_size
        self.init_message_dim = init_message_dim
        self.depth = depth
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.attached_feat = attach_feat

        assert dynamic_depth in [
            'none', 'truncnorm', 'uniform'
        ], 'If dynamic depth, it should be truncnorm or uniform'
        self.dynamic_depth = dynamic_depth

        self.dropout = nn.Dropout(p=dropout)
        if activation == 'relu':
            self.act_func = nn.ReLU()
        else:
            raise ValueError('Only ReLU activation function is supported')

        if self.input_layer == "fc":
            input_dim = self.init_message_dim
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        # Bug here, if input_layer is none, then w_h_input_size should have init_messages_dim
        # instead of hidden_size for first iteration of message passing alone (in subsequent
        # iterations, message size will be n_bonds x hidden_size but during first iteration,
        # message size will be init_messages_dim
        if self.attached_feat:
            w_h_input_size = hidden_size + attached_fea_fdim
        else:
            w_h_input_size = hidden_size

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
                ndepth = np.random.uniform(self.depth - 3, self.depth + 3)
            elif self.dynamic_depth == 'truncnorm':
                mu, sigma = self.depth, 1
                lower, upper = mu - 3 * sigma, mu + 3 * sigma
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma,
                                    loc=mu,
                                    scale=sigma)
                ndepth = int(X.rvs(1))
        else:
            ndepth = self.ndepth

        for _ in range(ndepth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            nei_message = _select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_feat:
                attached_nei_feats = _select_neighbor_and_aggregated(
                    attached_feats, a2attached)
                a_message = torch.cat((nei_message, attached_nei_feats), dim=1)

            if not self.atom_messages:
                rev_message = message[b2revb]
                if self.attached_feat:
                    atom_rev_message = attached_fea[b2a[b2revb]]
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
    """Generates attention head using GroverMPNEncoder for generating query, key and value"""

    def __init__(self,
                 hidden_size: int = 128,
                 bias: bool = True,
                 depth: int = 1,
                 dropout: float = 0.0,
                 undirected: bool = False,
                 dense: int = 1,
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
                                      dense=dense,
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
                                      dense=dense,
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
                                      dense=dense,
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
    """

    def __init__(self, atom_messages, input_dim, hidden_size, num_heads,
                 dropout, bias, activation):
        super(GroverMTBlock, self).__init__()
        self.atom_messages = atom_messages
        self.res_connection = res_connection
        self.num_heads = self.num_heads
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
                                    depth=depth,
                                    dropout=dropout,
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
        x_out = self.attn(queries, keys, values)
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.W_o(x_out)

        # support no residual connection in MTBlock.
        if self.res_connection:
            if self.atom_messages:
                f_atoms = self.sublayer(f_atoms, x_out)
            else:
                f_bonds = self.sublayer(f_bonds, x_out)
        batch = f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope,
        return batch


class GroverTransEncoder(nn.Module):
    """GroverTransEncoder for encoding a molecular graph

    The GroverTransEncoder layer is used for encoding a molecular graph.
    The layer can return four possible output depending on the `atom_emb_output`
    choice. If it `none`, it returns output from multihead attention block directly.
    If it is `atom`, then it aggregates node messages from node hidden states, it also aggregates
    node messages from incoming bond's hidden states to the nodes and returns two tensors.
    It it is `bond`, it aggregates bond embeddings from bond hidden states, it also aggregates another
    set of bond embeddings from bond's source nodes hidden states and returns two tensors.
    If it is `both`, it returns four tensor (`bond` option + `atom` option).

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
    num_attn_head: int
        the number of attention AttentionHead.
    atom_emb_output:  bool
        enable the output aggregation after message passing.
                                            atom_messages:      True                      False
        - "none": no aggregation         output size:     (num_atoms, hidden_size)    (num_bonds, hidden_size)
        -"atom":  aggregating to atom  output size:     (num_atoms, hidden_size)    (num_atoms, hidden_size)
        -"bond": aggragating to bond.   output size:     (num_bonds, hidden_size)    (num_bonds, hidden_size)
        -"both": aggregating to atom&bond. output size:  (num_atoms, hidden_size)    (num_bonds, hidden_size)
                                                        (num_bonds, hidden_size)    (num_atoms, hidden_size)
    bias: bool
        enable bias term in all linear layers.
    res_connection: bool
        enables the skip-connection in MTBlock.
    """
    super(GroverTransEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.edge_fdim = edge_fdim
    self.node_fdim = node_fdim
    self.depth = depth
    self.undirected = undirected
    self.dropout = dropout
    self.activation = activation
    self.num_mt_block = num_mt_block
    self.num_attn_head = num_attn_head
    self.atom_emb_output = atom_emb_output
    self.bias = bias
    self.res_connection = self.res_connection
