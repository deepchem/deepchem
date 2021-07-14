import math
import copy
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
except:
    raise ImportError('These classes require Torch to be installed.')

def clones(module, N):
    """Produce N identical layers.
    
    Parameters
    ----------
    N: int
      Number of identical layers to be returned.
    module: nn.Module
      The module where the N identical layers are to be added.
    
    Returns
    -------
    Torch module with N identical layers.
    """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def attention(query, key, value, adj_matrix, distances_matrix, edges_att,
              mask=None, dropout=None, 
              lambdas=(0.3, 0.3, 0.4), trainable_lambda=False,
              distance_matrix_kernel=None, use_edge_features=False, control_edges=False,
              eps=1e-6, inf=1e12):
    """Compute scaled dot product attention for the Molecular Attention Transformer [1]_.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Parameters
    ----------
    query: Tensor
      Query tensor for attention.
    key: Tensor
      Key magnitude for query.
    value: Tensor
      This will be multiplied with p_weights to determinethe atom_features matrix.
    adj_matrix: Tensor
      One of the outputs from the MATFeaturizer.
    distances_matrix: Tensor
      One of the outputs from the MATFeaturizer.
    edges_att: Tensor
      One of the outputs from the MATFeaturizer.
    mask: Tensor
      Mask for attention.
    dropout: float
      Dropout probability for neurons in the layer.
    lambdas: Tuple[float]
      Lambda values for attention.
    trainable_lambda: Bool
      If True, lambda parameters become trainable.
    distance_matrix_kernel: string
      Determines which kernel to use.
    use_edge_features: Bool
      If True, adjacency matrix becomes a View of edges_att.
    control_edges: Bool
      If True, controls edges.
    eps: float
      Default value of epsilon used.
    inf: float
      Default value of infinity used.
    
    Returns
    -------
    atoms_features: Tensor
      Atom Features Tensor. 
    p_weighted: Tensor
      Weights from attention with added dropout.
    p_attn: Tensor
      Softmax output of key, query scores.
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, -inf)
    p_attn = F.softmax(scores, dim = -1)

    if use_edge_features:
        adj_matrix = edges_att.view(adj_matrix.shape)

    # Prepare adjacency matrix
    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix
    
    p_dist = distances_matrix
    
    if trainable_lambda:
        softmax_attention, softmax_distance, softmax_adjacency = lambdas.cuda()
        p_weighted = softmax_attention * p_attn + softmax_distance * p_dist + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_distance, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_distance * p_dist + lambda_adjacency * p_adj
    
    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_features = torch.matmul(p_weighted, value)     
    return atoms_features, p_weighted, p_attn


class GraphTransformer(nn.Module):
    """Transformer for a molecular graph.

    Takes input, processes it through encoder, embedding and generator layers then returns output.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    """

    def __init__(self, encoder, src_embed, generator):
        """Initialize a graph transformer layer.

        Parameters
        ----------
        encoder: dc.layers.Encoder
        Encoder layer for the transformer.
        src_embed: dc.layers.Embeddings
        Embedding layer for the transformer.
        generator: dc.layers.Generator
        Generator layer for the transformer.
        """

        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
    
    def forward(self, src, src_mask, adj_matrix, distances_matrix, edges_att):
        """Process masked source and target sequences."""
        return self.predict(self.encode(src, src_mask, adj_matrix, distances_matrix, edges_att), src_mask)
    
    def encode(self, src, src_mask, adj_matrix, distances_matrix, edges_att):
        return self.encoder(self.src_embed(src), src_mask, adj_matrix, distances_matrix, edges_att)
    
    def predict(self, out, out_mask):
        return self.generator(out, out_mask)

class Generator(nn.Module):
    """Apply generation to input.
    
    The layer takes input and applies a combined step linear and softmax generation to it.
    
    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    """
    
    def __init__(self, d_model, aggregation_type='mean', n_output=1, n_layers=1, 
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False):
        """Initialize a generator layer.

        Parameters
        ----------
        d_model: Tuple[int] or int
          Input sample size.
        aggregation_type: Str
          Type of aggregation to be used.
          Can choose between mean, sum, dummy_node.
        n_output: Tuple[int] or int
          Output sample size.
        n_layers: int
          Number of generator Blocks.
        leaky_relu_slope: Float
          Negative slope magnitude for LeakyReLU.
        dropout: Float
          Dropout probability.
        scale_norm: Bool
          If True, applies ScaleNorm, else applies LayerNorm with input sample size d_model.
        """

        super(Generator, self).__init__()
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers-1):
                self.proj.append(nn.Linear(d_model, d_model))
                self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                self.proj.append(ScaleNorm(d_model) if scale_norm else LayerNorm(d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_model, n_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_avg_pooling = out_sum
        elif self.aggregation_type == 'dummy_node':
            out_avg_pooling = out_masked[:,0]
        projected = self.proj(out_avg_pooling)
        return projected

class PositionGenerator(nn.Module):
    """Simpler variant of the Generator class.

    It has the same function as the Generator class (combined linear and softmax generation step) but only uses LayerNorm and adds a Linear layer afterwards.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    """

    def __init__(self, d_model):
        """Initialize a position generator layer.

        Parameters
        ----------
        d_model: Tuple[int] or int
          Input sample size.
        """
        super(PositionGenerator, self).__init__()
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 3)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = self.norm(x) * mask
        projected = self.proj(out_masked)
        return projected

class LayerNorm(nn.Module):
    """Apply Layer Normalization to input.
    
    The layer takes input and applies layer Normalization to it.
    
    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """Apply Scale Normalization to input.
    
    All G values are initialized to sqrt(d).

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    """

    def __init__(self, scale, eps=1e-5):
        """Initialize a ScaleNorm layer.

        Parameters
        ----------
        scale: Tensor
          Scale magnitude.
        eps: float
          Epsilon value.
        """

        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps
        
    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm

class Encoder(nn.Module):
    """Encoder block for the Molecule Attention Transformer in [1]_.
    
    A stack of N layers which form the encoder block.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    """
    
    def __init__(self, layer, N, scale_norm):
        """Initialize an Encoder block.

        Parameters
        ----------
        layer: dc.layers
          Layer to be stacked in the encoder block.
        N: int
          Number of identical layers to be stacked.
        scale_norm: Bool
          If True, uses ScaleNorm, else uses LayerNorm.
        """

        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        if scale_norm:
            self.norm = ScaleNorm(layer.size)
        else:
            self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, distances_matrix, edges_att)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """SublayerConnection layer as used in [1]_.
    
    The SublayerConnection layer is a residual layer which is then passed through Layer Normalization.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    """
    
    def __init__(self, size, dropout, scale_norm):
        """Initialize a SublayerConnection Layer.

        Parameters
        ----------
        size: int
          Normalized shape of LayerNorm/ScaleNorm layer.
        dropout: float
          Dropout probability.
        scale_norm: Bool
          If True, uses ScaleNorm, else uses LayerNorm.
        """

        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """Encoder layer for use in the Molecular Attention Transformer.
    
    The Encoder layer is formed by adding self-attention and feed-forward to the encoder block.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    """

    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        """Initialize an Encoder layer.

        Parameters
        ----------
        size: int
          Normalized shape/length of ScaleNorm/LayerNorm layer.
        self_attn: Tensor
          The p_attn attribute from the attention function for the Molecular Attention Transformer.
        feed_forward: Bool
          If True, uses ScaleNorm, else uses LayerNorm.
        dropout:
          asd
        scale_norm: Bool
          If True, uses ScaleNorm, else uses LayerNorm.
        """

        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
        self.size = size

    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, distances_matrix, edges_att, mask))
        return self.sublayer[1](x, self.feed_forward)

class EdgeFeaturesLayer(nn.Module):
    def __init__(self, d_model, d_edge, h, dropout):
        super(EdgeFeaturesLayer, self).__init__()
        assert d_model % h == 0
        d_k = d_model // h
        self.linear = nn.Linear(d_edge, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)

    def forward(self, x):
        p_edge = x.permute(0, 2, 3, 1)
        p_edge = self.linear(p_edge).permute(0, 3, 1, 2)
        return torch.relu(p_edge)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, lambda_attention=0.3, lambda_distance=0.3, trainable_lambda=False, 
                 distance_matrix_kernel='softmax', use_edge_features=False, control_edges=False, integrated_distances=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.trainable_lambda = trainable_lambda
        if trainable_lambda:
            lambda_adjacency = 1. - lambda_attention - lambda_distance
            lambdas_tensor = torch.tensor([lambda_attention, lambda_distance, lambda_adjacency], requires_grad=True)
            self.lambdas = torch.nn.Parameter(lambdas_tensor)
        else:
            lambda_adjacency = 1. - lambda_attention - lambda_distance
            self.lambdas = (lambda_attention, lambda_distance, lambda_adjacency)
            
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        if distance_matrix_kernel == 'softmax':
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim = -1)
        elif distance_matrix_kernel == 'exp':
            self.distance_matrix_kernel = lambda x: torch.exp(-x)
        self.integrated_distances = integrated_distances
        self.use_edge_features = use_edge_features
        self.control_edges = control_edges
        if use_edge_features:
            d_edge = 11 if not integrated_distances else 12
            self.edges_feature_layer = EdgeFeaturesLayer(d_model, d_edge, h, dropout)
        
    def forward(self, query, key, value, adj_matrix, distances_matrix, edges_att, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        distances_matrix = distances_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
        distances_matrix = self.distance_matrix_kernel(distances_matrix)
        p_dist = distances_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        if self.use_edge_features:
            if self.integrated_distances:
                edges_att = torch.cat((edges_att, distances_matrix.unsqueeze(1)), dim=1)
            edges_att = self.edges_feature_layer(edges_att)
        
        x, self.attn, self.self_attn = attention(query, key, value, adj_matrix, 
                                                 p_dist, edges_att,
                                                 mask=mask, dropout=self.dropout,
                                                 lambdas=self.lambdas,
                                                 trainable_lambda=self.trainable_lambda,
                                                 distance_matrix_kernel=self.distance_matrix_kernel,
                                                 use_edge_features=self.use_edge_features,
                                                 control_edges=self.control_edges)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements the Feed-Forward algorithm for Molecular Attention Transformer."
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
            
    def forward(self, x):
        if self.N_dense == 0:
            return x
        
        for i in range(len(self.linears)-1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
            
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))

class Embeddings(nn.Module):
    """Implements Embedding for Molecular Attention Transformer."""
    def __init__(self, d_model, d_atom, dropout):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))