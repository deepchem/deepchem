from deepchem.models.torch_models.layers import SublayerConnection, PositionwiseFeedForward


class GroverMPNEncoder(nn.Module):
    """Performs Message Passing to encodes a molecule"""

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
        self.act_func = get_activation(activation)

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
    """Generates attention head using GroverMPNEncoder"""

    def __init__(self):
        pass

    def forward(self):
        pass


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
