import math
import numpy as np
from typing import Any, Tuple, Optional, Sequence, List, Union
from collections.abc import Sequence as SequenceCollection
try:
    import torch
    from torch import Tensor
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')

try:
    from torch_scatter import scatter_mean
except ModuleNotFoundError:
    pass

from deepchem.utils.typing import OneOrMany, ActivationFn, ArrayLike
from deepchem.utils.pytorch_utils import get_activation
from torch.nn import init as initializers


class MultilayerPerceptron(nn.Module):
    """A simple fully connected feed-forward network, otherwise known as a multilayer perceptron (MLP).

    Examples
    --------
    >>> model = MultilayerPerceptron(d_input=10, d_hidden=3, n_layers=2, d_output=2, dropout=0.0, activation_fn='relu')
    >>> x = torch.ones(2, 10)
    >>> out = model(x)
    >>> print(out.shape)
    torch.Size([2, 2])

    """

    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 n_layers: int,
                 d_output: int,
                 dropout: float = 0.0,
                 activation_fn: ActivationFn = 'relu'):
        """Initialize the model.

        Parameters
        ----------
        d_input: int
            the dimension of the input layer
        d_hidden: int
            the dimension of the hidden layers
        n_layers: int
            the number of hidden layers
        d_output: int
            the dimension of the output layer
        dropout: float
            the dropout probability
        activation_fn: str
            the activation function to use in the hidden layers
        """
        super(MultilayerPerceptron, self).__init__()
        self.input_layer = nn.Linear(d_input, d_hidden)
        self.hidden_layer = nn.Linear(d_hidden, d_hidden)
        self.output_layer = nn.Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.d_input = d_input
        self.d_output = d_output
        self.activation_fn = get_activation(activation_fn)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""

        if not self.n_layers:
            return x

        if self.n_layers == 1:
            x = self.input_layer(x)
            x = self.activation_fn(x)
            return x

        x = self.input_layer(x)
        x = self.activation_fn(x)
        for i in range(self.n_layers - 1):
            x = self.hidden_layer(x)
            x = self.dropout(x)
            x = self.activation_fn(x)
        x = self.output_layer(x)
        return x


class CNNModule(nn.Module):
    """A 1, 2, or 3 dimensional convolutional network for either regression or classification.
    The network consists of the following sequence of layers:
    - A configurable number of convolutional layers
    - A global pooling layer (either max pool or average pool)
    - A final fully connected layer to compute the output
    It optionally can compose the model from pre-activation residual blocks, as
    described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
    convolution layers.  This often leads to easier training, especially when using a
    large number of layers.  Note that residual blocks can only be used when
    successive layers have the same output shape.  Wherever the output shape changes, a
    simple convolution layer will be used even if residual=True.
    Examples
    --------
    >>> model = CNNModule(n_tasks=5, n_features=8, dims=2, layer_filters=[3,8,8,16], kernel_size=3, n_classes = 7, mode='classification', uncertainty=False, padding='same')
    >>> x = torch.ones(2, 224, 224, 8)
    >>> x = model(x)
    >>> for tensor in x:
    ...    print(tensor.shape)
    torch.Size([2, 5, 7])
    torch.Size([2, 5, 7])
    """

    def __init__(self,
                 n_tasks: int,
                 n_features: int,
                 dims: int,
                 layer_filters: List[int] = [100],
                 kernel_size: OneOrMany[int] = 5,
                 strides: OneOrMany[int] = 1,
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 dropouts: OneOrMany[float] = 0.5,
                 activation_fns: OneOrMany[ActivationFn] = 'relu',
                 pool_type: str = 'max',
                 mode: str = 'classification',
                 n_classes: int = 2,
                 uncertainty: bool = False,
                 residual: bool = False,
                 padding: Union[int, str] = 'valid') -> None:
        """Create a CNN.

        Parameters
        ----------
        n_tasks: int
            number of tasks
        n_features: int
            number of features
        dims: int
            the number of dimensions to apply convolutions over (1, 2, or 3)
        layer_filters: list
            the number of output filters for each convolutional layer in the network.
            The length of this list determines the number of layers.
        kernel_size: int, tuple, or list
            a list giving the shape of the convolutional kernel for each layer.  Each
            element may be either an int (use the same kernel width for every dimension)
            or a tuple (the kernel width along each dimension).  Alternatively this may
            be a single int or tuple instead of a list, in which case the same kernel
            shape is used for every layer.
        strides: int, tuple, or list
            a list giving the stride between applications of the  kernel for each layer.
            Each element may be either an int (use the same stride for every dimension)
            or a tuple (the stride along each dimension).  Alternatively this may be a
            single int or tuple instead of a list, in which case the same stride is
            used for every layer.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization
            of each layer.  The length of this list should equal len(layer_filters)+1,
            where the final element corresponds to the dense layer.  Alternatively this
            may be a single value instead of a list, in which case the same value is used
            for every layer.
        bias_init_consts: list or float
            the value to initialize the biases in each layer to.  The length of this
            list should equal len(layer_filters)+1, where the final element corresponds
            to the dense layer.  Alternatively this may be a single value instead of a
            list, in which case the same value is used for every layer.
        dropouts: list or float
            the dropout probability to use for each layer.  The length of this list should equal len(layer_filters).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer
        activation_fns: str or list
            the torch activation function to apply to each layer. The length of this list should equal
            len(layer_filters).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer, 'relu' by default
        pool_type: str
            the type of pooling layer to use, either 'max' or 'average'
        mode: str
            Either 'classification' or 'regression'
        n_classes: int
            the number of classes to predict (only used in classification mode)
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        residual: bool
            if True, the model will be composed of pre-activation residual blocks instead
            of a simple stack of convolutional layers.
        padding: str, int or tuple
            the padding to use for convolutional layers, either 'valid' or 'same'
        """

        super(CNNModule, self).__init__()

        if dims not in (1, 2, 3):
            raise ValueError('Number of dimensions must be 1, 2 or 3')

        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        self.n_tasks = n_tasks
        self.n_features = n_features
        self.dims = dims
        self.mode = mode
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        self.mode = mode
        self.layer_filters = layer_filters
        self.residual = residual

        n_layers = len(layer_filters)

        # PyTorch layers require input and output channels as parameter
        # if only one layer to make the model creating loop below work, multiply layer_filters wutg 2
        if len(layer_filters) == 1:
            layer_filters = layer_filters * 2

        if not isinstance(kernel_size, SequenceCollection):
            kernel_size = [kernel_size] * n_layers
        if not isinstance(strides, SequenceCollection):
            strides = [strides] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers

        self.activation_fns = [get_activation(f) for f in activation_fns]
        self.dropouts = dropouts

        if uncertainty:

            if mode != 'regression':
                raise ValueError(
                    "Uncertainty is only supported in regression mode")

            if any(d == 0.0 for d in dropouts):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty'
                )

        # Python tuples use 0 based indexing, dims defines number of dimension for convolutional operation
        ConvLayer = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[self.dims - 1]

        if pool_type == 'average':
            PoolLayer = (F.avg_pool1d, F.avg_pool2d,
                         F.avg_pool3d)[self.dims - 1]
        elif pool_type == 'max':
            PoolLayer = (F.max_pool1d, F.max_pool2d,
                         F.max_pool3d)[self.dims - 1]
        else:
            raise ValueError("pool_type must be either 'average' or 'max'")

        self.PoolLayer = PoolLayer
        self.layers = nn.ModuleList()

        in_shape = n_features

        for out_shape, size, stride, weight_stddev, bias_const in zip(
                layer_filters, kernel_size, strides, weight_init_stddevs,
                bias_init_consts):

            layer = ConvLayer(in_channels=in_shape,
                              out_channels=out_shape,
                              kernel_size=size,
                              stride=stride,
                              padding=padding,
                              dilation=1,
                              groups=1,
                              bias=True)

            nn.init.normal_(layer.weight, 0, weight_stddev)

            # initializing layer bias with nn.init gives mypy typecheck error
            # using the following workaround
            if layer.bias is not None:
                layer.bias = nn.Parameter(
                    torch.full(layer.bias.shape, bias_const))

            self.layers.append(layer)

            in_shape = out_shape

        self.classifier_ffn = nn.LazyLinear(self.n_tasks * self.n_classes)
        self.output_layer = nn.LazyLinear(self.n_tasks)
        self.uncertainty_layer = nn.LazyLinear(self.n_tasks)

    def forward(self, inputs: OneOrMany[torch.Tensor]) -> List[Any]:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input Tensor
        Returns
        -------
        torch.Tensor
            Output as per use case : regression/classification
        """
        if isinstance(inputs, torch.Tensor):
            x, dropout_switch = inputs, None
        else:
            x, dropout_switch = inputs

        x = torch.transpose(x, 1, -1)  # n h w c -> n c h w

        prev_layer = x

        for layer, activation_fn, dropout in zip(self.layers,
                                                 self.activation_fns,
                                                 self.dropouts):
            x = layer(x)

            if dropout > 0. and dropout_switch:
                x = F.dropout(x, dropout)

            # residual blocks can only be used when successive layers have the same output shape
            if self.residual and x.shape[1] == prev_layer.shape[1]:
                x = x + prev_layer

            if activation_fn is not None:
                x = activation_fn(x)

            prev_layer = x

        x = self.PoolLayer(x, kernel_size=x.size()[2:])

        outputs = []
        batch_size = x.shape[0]

        x = torch.reshape(x, (batch_size, -1))

        if self.mode == "classification":

            logits = self.classifier_ffn(x)
            logits = logits.view(batch_size, self.n_tasks, self.n_classes)
            output = F.softmax(logits, dim=2)
            outputs = [output, logits]

        else:
            output = self.output_layer(x)
            output = output.view(batch_size, self.n_tasks)

            if self.uncertainty:
                log_var = self.uncertainty_layer(x)
                log_var = log_var.view(batch_size, self.n_tasks, 1)
                var = torch.exp(log_var)
                outputs = [output, var, output, log_var]
            else:
                outputs = [output]

        return outputs


class ScaleNorm(nn.Module):
    """Apply Scale Normalization to input.

    The ScaleNorm layer first computes the square root of the scale, then computes the matrix/vector norm of the input tensor.
    The norm value is calculated as `sqrt(scale) / matrix norm`.
    Finally, the result is returned as `input_tensor * norm value`.

    This layer can be used instead of LayerNorm when a scaled version of the norm is required.
    Instead of performing the scaling operation (`scale / norm`) in a lambda-like layer, we are defining it within this layer to make prototyping more efficient.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import ScaleNorm
    >>> scale = 0.35
    >>> layer = ScaleNorm(scale)
    >>> input_tensor = torch.tensor([[1.269, 39.36], [0.00918, -9.12]])
    >>> output_tensor = layer(input_tensor)
    """

    def __init__(self, scale: float, eps: float = 1e-5):
        """Initialize a ScaleNorm layer.

        Parameters
        ----------
        scale: float
            Scale magnitude.
        eps: float
            Epsilon value. Default = 1e-5.
        """
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.scale / torch.norm(x, dim=-1,
                                       keepdim=True).clamp(min=self.eps)
        return x * norm


class MultiHeadedMATAttention(nn.Module):
    """First constructs an attention layer tailored to the Molecular Attention Transformer [1]_ and then converts it into Multi-Headed Attention.

    In Multi-Headed attention the attention mechanism multiple times parallely through the multiple attention heads.
    Thus, different subsequences of a given sequences can be processed differently.
    The query, key and value parameters are split multiple ways and each split is passed separately through a different attention head.
    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    Examples
    --------
    >>> from deepchem.models.torch_models.layers import MultiHeadedMATAttention, MATEmbedding
    >>> import deepchem as dc
    >>> import torch
    >>> input_smile = "CC"
    >>> feat = dc.feat.MATFeaturizer()
    >>> input_smile = "CC"
    >>> out = feat.featurize(input_smile)
    >>> node = torch.tensor(out[0].node_features).float().unsqueeze(0)
    >>> adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
    >>> dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
    >>> mask = torch.sum(torch.abs(node), dim=-1) != 0
    >>> layer = MultiHeadedMATAttention(
    ...    dist_kernel='softmax',
    ...    lambda_attention=0.33,
    ...    lambda_distance=0.33,
    ...    h=16,
    ...    hsize=1024,
    ...    dropout_p=0.0)
    >>> op = MATEmbedding()(node)
    >>> output = layer(op, op, op, mask, adj, dist)
    """

    def __init__(self,
                 dist_kernel: str = 'softmax',
                 lambda_attention: float = 0.33,
                 lambda_distance: float = 0.33,
                 h: int = 16,
                 hsize: int = 1024,
                 dropout_p: float = 0.0,
                 output_bias: bool = True):
        """Initialize a multi-headed attention layer.
        Parameters
        ----------
        dist_kernel: str
            Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential.
        lambda_attention: float
            Constant to be multiplied with the attention matrix.
        lambda_distance: float
            Constant to be multiplied with the distance matrix.
        h: int
            Number of attention heads.
        hsize: int
            Size of dense layer.
        dropout_p: float
            Dropout probability.
        output_bias: bool
            If True, dense layers will use bias vectors.
        """
        super().__init__()
        if dist_kernel == "softmax":
            self.dist_kernel = lambda x: torch.softmax(-x, dim=-1)
        elif dist_kernel == "exp":
            self.dist_kernel = lambda x: torch.exp(-x)
        self.lambda_attention = lambda_attention
        self.lambda_distance = lambda_distance
        self.lambda_adjacency = 1.0 - self.lambda_attention - self.lambda_distance
        self.d_k = hsize // h
        self.h = h
        linear_layer = nn.Linear(hsize, hsize)
        self.linear_layers = nn.ModuleList([linear_layer for _ in range(3)])
        self.dropout_p = nn.Dropout(dropout_p)
        self.output_linear = nn.Linear(hsize, hsize, output_bias)

    def _single_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor,
            adj_matrix: torch.Tensor,
            distance_matrix: torch.Tensor,
            dropout_p: float = 0.0,
            eps: float = 1e-6,
            inf: float = 1e12) -> Tuple[torch.Tensor, torch.Tensor]:
        """Defining and computing output for a single MAT attention layer.
        Parameters
        ----------
        query: torch.Tensor
            Standard query parameter for attention.
        key: torch.Tensor
            Standard key parameter for attention.
        value: torch.Tensor
            Standard value parameter for attention.
        mask: torch.Tensor
            Masks out padding values so that they are not taken into account when computing the attention score.
        adj_matrix: torch.Tensor
            Adjacency matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dist_matrix: torch.Tensor
            Distance matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dropout_p: float
            Dropout probability.
        eps: float
            Epsilon value
        inf: float
            Value of infinity to be used.
        """
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2],
                                         1) == 0, -inf)
        p_attn = F.softmax(scores, dim=-1)

        adj_matrix = adj_matrix / (
            torch.sum(torch.tensor(adj_matrix), dim=-1).unsqueeze(2) + eps)

        if len(adj_matrix.shape) <= 3:
            p_adj = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
        else:
            p_adj = adj_matrix.repeat(1, query.shape[1], 1, 1)

        distance_matrix = torch.tensor(distance_matrix).squeeze().masked_fill(
            mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)

        distance_matrix = self.dist_kernel(distance_matrix)

        p_dist = distance_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        p_weighted = self.lambda_attention * p_attn + self.lambda_distance * p_dist + self.lambda_adjacency * p_adj
        p_weighted = self.dropout_p(p_weighted)

        return torch.matmul(p_weighted.float(), value.float()), p_attn

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor,
                adj_matrix: torch.Tensor,
                distance_matrix: torch.Tensor,
                dropout_p: float = 0.0,
                eps: float = 1e-6,
                inf: float = 1e12) -> torch.Tensor:
        """Output computation for the MultiHeadedAttention layer.
        Parameters
        ----------
        query: torch.Tensor
            Standard query parameter for attention.
        key: torch.Tensor
            Standard key parameter for attention.
        value: torch.Tensor
            Standard value parameter for attention.
        mask: torch.Tensor
            Masks out padding values so that they are not taken into account when computing the attention score.
        adj_matrix: torch.Tensor
            Adjacency matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dist_matrix: torch.Tensor
            Distance matrix of the input molecule, returned from dc.feat.MATFeaturizer()
        dropout_p: float
            Dropout probability.
        eps: float
            Epsilon value
        inf: float
            Value of infinity to be used.
        """
        if mask is not None and len(mask.shape) <= 2:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = [
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        x, _ = self._single_attention(query, key, value, mask, adj_matrix,
                                      distance_matrix, dropout_p, eps, inf)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.h * self.d_k)

        return self.output_linear(x)


class MATEncoderLayer(nn.Module):
    """Encoder layer for use in the Molecular Attention Transformer [1]_.

    The MATEncoder layer primarily consists of a self-attention layer (MultiHeadedMATAttention) and a feed-forward layer (PositionwiseFeedForward).
    This layer can be stacked multiple times to form an encoder.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from rdkit import Chem
    >>> import torch
    >>> import deepchem
    >>> from deepchem.models.torch_models.layers import MATEmbedding, MATEncoderLayer
    >>> input_smile = "CC"
    >>> feat = deepchem.feat.MATFeaturizer()
    >>> out = feat.featurize(input_smile)
    >>> node = torch.tensor(out[0].node_features).float().unsqueeze(0)
    >>> adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
    >>> dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
    >>> mask = torch.sum(torch.abs(node), dim=-1) != 0
    >>> layer = MATEncoderLayer()
    >>> op = MATEmbedding()(node)
    >>> output = layer(op, mask, adj, dist)
    """

    def __init__(self,
                 dist_kernel: str = 'softmax',
                 lambda_attention: float = 0.33,
                 lambda_distance: float = 0.33,
                 h: int = 16,
                 sa_hsize: int = 1024,
                 sa_dropout_p: float = 0.0,
                 output_bias: bool = True,
                 d_input: int = 1024,
                 d_hidden: int = 1024,
                 d_output: int = 1024,
                 activation: str = 'leakyrelu',
                 n_layers: int = 1,
                 ff_dropout_p: float = 0.0,
                 encoder_hsize: int = 1024,
                 encoder_dropout_p: float = 0.0):
        """Initialize a MATEncoder layer.

        Parameters
        ----------
        dist_kernel: str
            Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential, for the self-attention layer.
        lambda_attention: float
            Constant to be multiplied with the attention matrix in the self-attention layer.
        lambda_distance: float
            Constant to be multiplied with the distance matrix in the self-attention layer.
        h: int
            Number of attention heads for the self-attention layer.
        sa_hsize: int
            Size of dense layer in the self-attention layer.
        sa_dropout_p: float
            Dropout probability for the self-attention layer.
        output_bias: bool
            If True, dense layers will use bias vectors in the self-attention layer.
        d_input: int
            Size of input layer in the feed-forward layer.
        d_hidden: int
            Size of hidden layer in the feed-forward layer.
        d_output: int
            Size of output layer in the feed-forward layer.
        activation: str
            Activation function to be used in the feed-forward layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
        n_layers: int
            Number of layers in the feed-forward layer.
        dropout_p: float
            Dropout probability in the feeed-forward layer.
        encoder_hsize: int
            Size of Dense layer for the encoder itself.
        encoder_dropout_p: float
            Dropout probability for connections in the encoder layer.
        """
        super(MATEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedMATAttention(dist_kernel, lambda_attention,
                                                 lambda_distance, h, sa_hsize,
                                                 sa_dropout_p, output_bias)
        self.feed_forward = PositionwiseFeedForward(d_input, d_hidden, d_output,
                                                    activation, n_layers,
                                                    ff_dropout_p)
        layer = SublayerConnection(size=encoder_hsize,
                                   dropout_p=encoder_dropout_p)
        self.sublayer = nn.ModuleList([layer for _ in range(2)])
        self.size = encoder_hsize

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                adj_matrix: torch.Tensor,
                distance_matrix: torch.Tensor,
                sa_dropout_p: float = 0.0) -> torch.Tensor:
        """Output computation for the MATEncoder layer.

        In the MATEncoderLayer intialization, self.sublayer is defined as an nn.ModuleList of 2 layers. We will be passing our computation through these layers sequentially.
        nn.ModuleList is subscriptable and thus we can access it as self.sublayer[0], for example.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        mask: torch.Tensor
            Masks out padding values so that they are not taken into account when computing the attention score.
        adj_matrix: torch.Tensor
            Adjacency matrix of a molecule.
        distance_matrix: torch.Tensor
            Distance matrix of a molecule.
        sa_dropout_p: float
            Dropout probability for the self-attention layer (MultiHeadedMATAttention).
        """
        x = self.sublayer[0](x,
                             self.self_attn(x,
                                            x,
                                            x,
                                            mask=mask,
                                            dropout_p=sa_dropout_p,
                                            adj_matrix=adj_matrix,
                                            distance_matrix=distance_matrix))
        return self.sublayer[1](x, self.feed_forward(x))


class SublayerConnection(nn.Module):
    """SublayerConnection layer which establishes a residual connection, as used in the Molecular Attention Transformer [1]_.

    The SublayerConnection layer is a residual layer which is then passed through Layer Normalization.
    The residual connection is established by computing the dropout-adjusted layer output of a normalized tensor and adding this to the original input tensor.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import SublayerConnection
    >>> scale = 0.35
    >>> layer = SublayerConnection(2, 0.)
    >>> input_ar = torch.tensor([[1., 2.], [5., 6.]])
    >>> output = layer(input_ar, input_ar)
  """

    def __init__(self, size: int, dropout_p: float = 0.0):
        """Initialize a SublayerConnection Layer.

        Parameters
        ----------
        size: int
            Size of layer.
        dropout_p: float
            Dropout probability.
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout_p = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Output computation for the SublayerConnection layer.

        Takes an input tensor x, then adds the dropout-adjusted sublayer output for normalized x to it.
        This is done to add a residual connection followed by LayerNorm.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        output: torch.Tensor
            Layer whose normalized output will be added to x.
        """
        if x is None:
            return self.dropout(self.norm(output))
        return x + self.dropout_p(self.norm(output))


class PositionwiseFeedForward(nn.Module):
    """PositionwiseFeedForward is a layer used to define the position-wise feed-forward (FFN) algorithm for the Molecular Attention Transformer [1]_

    Each layer in the MAT encoder contains a fully connected feed-forward network which applies two linear transformations and the given activation function.
    This is done in addition to the SublayerConnection module.

    Note: This modified version of `PositionwiseFeedForward` class contains `dropout_at_input_no_act` condition to facilitate its use in defining
        the feed-forward (FFN) algorithm for the Directed Message Passing Neural Network (D-MPNN) [2]_

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    .. [2] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import PositionwiseFeedForward
    >>> feed_fwd_layer = PositionwiseFeedForward(d_input = 2, d_hidden = 2, d_output = 2, activation = 'relu', n_layers = 1, dropout_p = 0.1)
    >>> input_tensor = torch.tensor([[1., 2.], [5., 6.]])
    >>> output_tensor = feed_fwd_layer(input_tensor)
  """

    def __init__(self,
                 d_input: int = 1024,
                 d_hidden: int = 1024,
                 d_output: int = 1024,
                 activation: str = 'leakyrelu',
                 n_layers: int = 1,
                 dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = False):
        """Initialize a PositionwiseFeedForward layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_hidden: int (same as d_input if d_output = 0)
            Size of hidden layer.
        d_output: int (same as d_input if d_output = 0)
            Size of output layer.
        activation: str
            Activation function to be used. Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
        n_layers: int
            Number of layers.
        dropout_p: float
            Dropout probability.
        dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
        """
        super(PositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act: bool = dropout_at_input_no_act

        if activation == 'relu':
            self.activation: Any = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        elif activation == "linear":
            self.activation = lambda x: x

        self.n_layers: int = n_layers
        d_output = d_output if d_output != 0 else d_input
        d_hidden = d_hidden if d_hidden != 0 else d_input

        if n_layers == 1:
            self.linears: Any = [nn.Linear(d_input, d_output)]

        else:
            self.linears = [nn.Linear(d_input, d_hidden)] + \
                            [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 2)] + \
                            [nn.Linear(d_hidden, d_output)]

        self.linears = nn.ModuleList(self.linears)
        dropout_layer = nn.Dropout(dropout_p)
        self.dropout_p = nn.ModuleList([dropout_layer for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output Computation for the PositionwiseFeedForward layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        if not self.n_layers:
            return x

        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return self.linears[0](self.dropout_p[0](x))
            else:
                return self.dropout_p[0](self.activation(self.linears[0](x)))

        else:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[0](x)
            for i in range(self.n_layers - 1):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))
            return self.linears[-1](x)


class MATEmbedding(nn.Module):
    """Embedding layer to create embedding for inputs.

    In an embedding layer, input is taken and converted to a vector representation for each input.
    In the MATEmbedding layer, an input tensor is processed through a dropout-adjusted linear layer and the resultant vector is returned.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import MATEmbedding
    >>> layer = MATEmbedding(d_input = 3, d_output = 3, dropout_p = 0.2)
    >>> input_tensor = torch.tensor([1., 2., 3.])
    >>> output = layer(input_tensor)
    """

    def __init__(self,
                 d_input: int = 36,
                 d_output: int = 1024,
                 dropout_p: float = 0.0):
        """Initialize a MATEmbedding layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_output: int
            Size of output layer.
        dropout_p: float
            Dropout probability for layer.
        """
        super(MATEmbedding, self).__init__()
        self.linear_unit = nn.Linear(d_input, d_output)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computation for the MATEmbedding layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor to be converted into a vector.
        """
        return self.dropout(self.linear_unit(x))


class MATGenerator(nn.Module):
    """MATGenerator defines the linear and softmax generator step for the Molecular Attention Transformer [1]_.

    In the MATGenerator, a Generator is defined which performs the Linear + Softmax generation step.
    Depending on the type of aggregation selected, the attention output layer performs different operations.

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import MATGenerator
    >>> layer = MATGenerator(hsize = 3, aggregation_type = 'mean', d_output = 1, n_layers = 1, dropout_p = 0.3, attn_hidden = 128, attn_out = 4)
    >>> input_tensor = torch.tensor([1., 2., 3.])
    >>> mask = torch.tensor([1., 1., 1.])
    >>> output = layer(input_tensor, mask)
    """

    def __init__(self,
                 hsize: int = 1024,
                 aggregation_type: str = 'mean',
                 d_output: int = 1,
                 n_layers: int = 1,
                 dropout_p: float = 0.0,
                 attn_hidden: int = 128,
                 attn_out: int = 4):
        """Initialize a MATGenerator.

        Parameters
        ----------
        hsize: int
            Size of input layer.
        aggregation_type: str
            Type of aggregation to be used. Can be 'grover', 'mean' or 'contextual'.
        d_output: int
            Size of output layer.
        n_layers: int
            Number of layers in MATGenerator.
        dropout_p: float
            Dropout probability for layer.
        attn_hidden: int
            Size of hidden attention layer.
        attn_out: int
            Size of output attention layer.
        """
        super(MATGenerator, self).__init__()

        if aggregation_type == 'grover':
            self.att_net = nn.Sequential(
                nn.Linear(hsize, attn_hidden, bias=False),
                nn.Tanh(),
                nn.Linear(attn_hidden, attn_out, bias=False),
            )
            hsize *= attn_out

        if n_layers == 1:
            self.proj: Any = nn.Linear(hsize, d_output)

        else:
            self.proj = []

            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(hsize, attn_hidden))
                self.proj.append(nn.LeakyReLU(negative_slope=0.1))
                self.proj.append(nn.LayerNorm(attn_hidden))
                self.proj.append(nn.Dropout(dropout_p))
            self.proj.append(nn.Linear(attn_hidden, d_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computation for the MATGenerator layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        mask: torch.Tensor
            Mask for padding so that padded values do not get included in attention score calculation.
        """

        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask

        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum

        elif self.aggregation_type == 'grover':
            out_attn = self.att_net(out_masked)
            out_attn = out_attn.masked_fill(mask == 0, -1e9)
            out_attn = F.softmax(out_attn, dim=1)
            out_avg_pooling = torch.matmul(torch.transpose(out_attn, -1, -2),
                                           out_masked)
            out_avg_pooling = out_avg_pooling.view(out_avg_pooling.size(0), -1)

        elif self.aggregation_type == 'contextual':
            out_avg_pooling = x

        return self.proj(out_avg_pooling)


class GraphNetwork(torch.nn.Module):
    """Graph Networks

    A Graph Network [1]_ takes a graph as input and returns an updated graph
    as output. The output graph has same structure as input graph but it
    has updated node features, edge features and global state features.

    Parameters
    ----------
    n_node_features: int
        Number of features in a node
    n_edge_features: int
        Number of features in a edge
    n_global_features: int
        Number of global features
    is_undirected: bool, optional (default True)
        Directed or undirected graph
    residual_connection: bool, optional (default True)
        If True, the layer uses a residual connection during training

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.layers import GraphNetwork as GN
    >>> n_nodes, n_node_features = 5, 10
    >>> n_edges, n_edge_features = 5, 2
    >>> n_global_features = 4
    >>> node_features = torch.randn(n_nodes, n_node_features)
    >>> edge_features = torch.randn(n_edges, n_edge_features)
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
    >>> global_features = torch.randn(1, n_global_features)
    >>> gn = GN(n_node_features=n_node_features, n_edge_features=n_edge_features, n_global_features=n_global_features)
    >>> node_features, edge_features, global_features = gn(node_features, edge_index, edge_features, global_features)

    References
    ----------
    .. [1] Battaglia et al, Relational inductive biases, deep learning, and graph networks. https://arxiv.org/abs/1806.01261 (2018)
  """

    def __init__(self,
                 n_node_features: int = 32,
                 n_edge_features: int = 32,
                 n_global_features: int = 32,
                 is_undirected: bool = True,
                 residual_connection: bool = True):
        super().__init__()
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_global_features = n_global_features
        self.is_undirected = is_undirected
        self.residual_connection = residual_connection

        self.edge_models, self.node_models, self.global_models = torch.nn.ModuleList(
        ), torch.nn.ModuleList(), torch.nn.ModuleList()
        self.edge_models.append(
            nn.Linear(in_features=n_node_features * 2 + n_edge_features +
                      n_global_features,
                      out_features=32))
        self.node_models.append(
            nn.Linear(in_features=n_node_features + n_edge_features +
                      n_global_features,
                      out_features=32))
        self.global_models.append(
            nn.Linear(in_features=n_node_features + n_edge_features +
                      n_global_features,
                      out_features=32))

        # Used for converting edges back to their original shape
        self.edge_dense = nn.Linear(in_features=32,
                                    out_features=n_edge_features)
        self.node_dense = nn.Linear(in_features=32,
                                    out_features=n_node_features)
        self.global_dense = nn.Linear(in_features=32,
                                      out_features=n_global_features)

    def reset_parameters(self) -> None:
        self.edge_dense.reset_parameters()
        self.node_dense.reset_parameters()
        self.global_dense.reset_parameters()
        for i in range(0, len(self.edge_models)):
            self.edge_models[i].reset_parameters()
        for i in range(0, len(self.node_models)):
            self.node_models[i].reset_parameters()
        for i in range(0, len(self.global_models)):
            self.global_models[i].reset_parameters()

    def _update_edge_features(self, node_features, edge_index, edge_features,
                              global_features, batch):
        src_index, dst_index = edge_index
        out = torch.cat((node_features[src_index], node_features[dst_index],
                         edge_features, global_features[batch]),
                        dim=1)
        assert out.shape[
            1] == self.n_node_features * 2 + self.n_edge_features + self.n_global_features
        for model in self.edge_models:
            out = model(out)
        return self.edge_dense(out)

    def _update_node_features(self, node_features, edge_index, edge_features,
                              global_features, batch):

        src_index, dst_index = edge_index
        # Compute mean edge features for each node by dst_index (each node
        # receives information from edges which have that node as its destination,
        # hence the computation uses dst_index to aggregate information)
        edge_features_mean_by_node = scatter_mean(edge_features,
                                                  dst_index,
                                                  dim=0)
        out = torch.cat(
            (node_features, edge_features_mean_by_node, global_features[batch]),
            dim=1)
        for model in self.node_models:
            out = model(out)
        return self.node_dense(out)

    def _update_global_features(self, node_features, edge_features,
                                global_features, node_batch_map,
                                edge_batch_map):
        edge_features_mean = scatter_mean(edge_features, edge_batch_map, dim=0)
        node_features_mean = scatter_mean(node_features, node_batch_map, dim=0)
        out = torch.cat(
            (edge_features_mean, node_features_mean, global_features), dim=1)
        for model in self.global_models:
            out = model(out)
        return self.global_dense(out)

    def forward(
            self,
            node_features: Tensor,
            edge_index: Tensor,
            edge_features: Tensor,
            global_features: Tensor,
            batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Output computation for a GraphNetwork

        Parameters
        ----------
        node_features: torch.Tensor
            Input node features of shape :math:`(|\mathcal{V}|, F_n)`
        edge_index: torch.Tensor
            Edge indexes of shape :math:`(2, |\mathcal{E}|)`
        edge_features: torch.Tensor
            Edge features of the graph, shape: :math:`(|\mathcal{E}|, F_e)`
        global_features: torch.Tensor
            Global features of the graph, shape: :math:`(F_g, 1)` where, :math:`|\mathcal{V}|` and :math:`|\mathcal{E}|` denotes the number of nodes and edges in the graph, :math:`F_n`, :math:`F_e`, :math:`F_g` denotes the number of node features, edge features and global state features respectively.
        batch: torch.LongTensor (optional, default: None)
            A vector that maps each node to its respective graph identifier. The attribute is used only when more than one graph are batched together during a single forward pass.
        """
        if batch is None:
            batch = node_features.new_zeros(node_features.size(0),
                                            dtype=torch.int64)

        node_features_copy, edge_features_copy, global_features_copy = node_features, edge_features, global_features
        if self.is_undirected is True:
            # holding bi-directional edges in case of undirected graphs
            edge_index = torch.cat((edge_index, edge_index.flip([0])), dim=1)
            edge_features_len = edge_features.shape[0]
            edge_features = torch.cat((edge_features, edge_features), dim=0)
        edge_batch_map = batch[edge_index[0]]
        edge_features = self._update_edge_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features,
                                                   edge_batch_map)
        node_features = self._update_node_features(node_features, edge_index,
                                                   edge_features,
                                                   global_features, batch)
        global_features = self._update_global_features(node_features,
                                                       edge_features,
                                                       global_features, batch,
                                                       edge_batch_map)

        if self.is_undirected is True:
            # coonverting edge features to its original shape
            split = torch.split(edge_features,
                                [edge_features_len, edge_features_len])
            edge_features = (split[0] + split[1]) / 2

        if self.residual_connection:
            edge_features += edge_features_copy
            node_features += node_features_copy
            global_features += global_features_copy

        return node_features, edge_features, global_features

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(n_node_features={self.n_node_features}, n_edge_features={self.n_edge_features}, n_global_features={self.n_global_features}, is_undirected={self.is_undirected}, residual_connection={self.residual_connection})'
        )


class Affine(nn.Module):
    """Class which performs the Affine transformation.

    This transformation is based on the affinity of the base distribution with
    the target distribution. A geometric transformation is applied where
    the parameters performs changes on the scale and shift of a function
    (inputs).

    Normalizing Flow transformations must be bijective in order to compute
    the logarithm of jacobian's determinant. For this reason, transformations
    must perform a forward and inverse pass.

    Example
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.layers import Affine
    >>> import torch
    >>> from torch.distributions import MultivariateNormal
    >>> # initialize the transformation layer's parameters
    >>> dim = 2
    >>> samples = 96
    >>> transforms = Affine(dim)
    >>> # forward pass based on a given distribution
    >>> distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> input = distribution.sample(torch.Size((samples, dim)))
    >>> len(transforms.forward(input))
    2
    >>> # inverse pass based on a distribution
    >>> len(transforms.inverse(input))
    2

    """

    def __init__(self, dim: int) -> None:
        """Create a Affine transform layer.

        Parameters
        ----------
        dim: int
            Value of the Nth dimension of the dataset.

        """

        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.zeros(self.dim))
        self.shift = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a transformation between two different distributions. This
        particular transformation represents the following function:
        y = x * exp(a) + b, where a is scale parameter and b performs a shift.
        This class also returns the logarithm of the jacobians determinant
        which is useful when invert a transformation and compute the
        probability of the transformation.

        Parameters
        ----------
        x : Sequence
            Tensor sample with the initial distribution data which will pass into
            the normalizing flow algorithm.

        Returns
        -------
        y : torch.Tensor
            Transformed tensor according to Affine layer with the shape of 'x'.
        log_det_jacobian : torch.Tensor
            Tensor which represents the info about the deviation of the initial
            and target distribution.

        """

        y = torch.exp(self.scale) * x + self.shift
        det_jacobian = torch.exp(self.scale.sum())
        log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

        return y, log_det_jacobian

    def inverse(self, y: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a transformation between two different distributions.
        This transformation represents the bacward pass of the function
        mention before. Its mathematical representation is x = (y - b) / exp(a)
        , where "a" is scale parameter and "b" performs a shift. This class
        also returns the logarithm of the jacobians determinant which is
        useful when invert a transformation and compute the probability of
        the transformation.

        Parameters
        ----------
        y : Sequence
            Tensor sample with transformed distribution data which will be used in
            the normalizing algorithm inverse pass.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor according to Affine layer with the shape of 'y'.
        inverse_log_det_jacobian : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.

        """

        x = (y - self.shift) / torch.exp(self.scale)
        det_jacobian = 1 / torch.exp(self.scale.sum())
        inverse_log_det_jacobian = torch.ones(
            x.shape[0]) * torch.log(det_jacobian)

        return x, inverse_log_det_jacobian


class DMPNNEncoderLayer(nn.Module):
    """
    Encoder layer for use in the Directed Message Passing Neural Network (D-MPNN) [1]_.

    The role of the DMPNNEncoderLayer class is to generate molecule encodings in following steps:

    - Message passing phase
    - Get new atom hidden states and readout phase
    - Concatenate the global features


    Let the diagram given below represent a molecule containing 5 atoms (nodes) and 4 bonds (edges):-

    |   1 --- 5
    |   |
    |   2 --- 4
    |   |
    |   3

    Let the bonds from atoms 1->2 (**B[12]**) and 2->1 (**B[21]**) be considered as 2 different bonds.
    Hence, by considering the same for all atoms, the total number of bonds = 8.

    Let:

    - **atom features** : ``a1, a2, a3, a4, a5``
    - **hidden states of atoms** : ``h1, h2, h3, h4, h5``
    - **bond features bonds** : ``b12, b21, b23, b32, b24, b42, b15, b51``
    - **initial hidden states of bonds** : ``(0)h12, (0)h21, (0)h23, (0)h32, (0)h24, (0)h42, (0)h15, (0)h51``

    The hidden state of every bond is a function of the concatenated feature vector which contains
    concatenation of the **features of initial atom of the bond** and **bond features**.

    Example: ``(0)h21 = func1(concat(a2, b21))``

    .. note::
     Here func1 is ``self.W_i``

    **The Message passing phase**

    The goal of the message-passing phase is to generate **hidden states of all the atoms in the molecule**.

    The hidden state of an atom is **a function of concatenation of atom features and messages (at T depth)**.

    A message is a sum of **hidden states of bonds coming to the atom (at T depth)**.

    .. note::
     Depth refers to the number of iterations in the message passing phase (here, T iterations). After each iteration, the hidden states of the bonds are updated.


    Example:
    ``h1 = func3(concat(a1, m1))``

    .. note::
     Here func3 is ``self.W_o``.

     `m1` refers to the message coming to the atom.

    ``m1 = (T-1)h21 + (T-1)h51``
    (hidden state of bond 2->1 + hidden state of bond 5->1) (at T depth)

    for, depth T = 2:

    - the hidden states of the bonds @ 1st iteration will be => (0)h21, (0)h51
    - the hidden states of the bonds @ 2nd iteration will be => (1)h21, (1)h51

    The hidden states of the bonds in 1st iteration are already know.
    For hidden states of the bonds in 2nd iteration, we follow the criterion that:

    - hidden state of the bond is a function of **initial hidden state of bond**
    and **messages coming to that bond in that iteration**

    Example:
    ``(1)h21 = func2( (0)h21 , (1)m21 )``

    .. note::
     Here func2 is ``self.W_h``.

     `(1)m21` refers to the messages coming to that bond 2->1 in that 2nd iteration.

    Messages coming to a bond in an iteration is **a sum of hidden states of bonds (from previous iteration) coming to this bond**.

    Example:
    ``(1)m21 = (0)h32 + (0)h42``

    |   2 <--- 3
    |   ^
    |   |
    |   4

    **Computing the messages**

    .. code-block:: python

                             B0      B1      B2      B3      B4      B5      B6      B7      B8
        f_ini_atoms_bonds = [(0)h12, (0)h21, (0)h23, (0)h32, (0)h24, (0)h42, (0)h15, (0)h51, h(-1)]


    .. note::
     h(-1) is an empty array of the same size as other hidden states of bond states.

    .. code-block:: python

                    B0      B1      B2      B3      B4      B5      B6      B7       B8
        mapping = [ [-1,B7] [B3,B5] [B0,B5] [-1,-1] [B0,B3] [-1,-1] [B1,-1] [-1,-1]  [-1,-1] ]

    Later, the encoder will map the concatenated features from the ``f_ini_atoms_bonds``
    to ``mapping`` in each iteration upto Tth iteration.

    Next the encoder will sum-up the concat features within same bond index.

    .. code-block:: python

                        (1)m12           (1)m21           (1)m23              (1)m32          (1)m24           (1)m42           (1)m15          (1)m51            m(-1)
        message = [ [h(-1) + (0)h51] [(0)h32 + (0)h42] [(0)h12 + (0)h42] [h(-1) + h(-1)] [(0)h12 + (0)h32] [h(-1) + h(-1)] [(0)h21 + h(-1)] [h(-1) + h(-1)]  [h(-1) + h(-1)] ]

    Hence, this is how encoder can get messages for message-passing steps.

    **Get new atom hidden states and readout phase**

    Hence now for h1:

    .. code-block:: python

        h1 = func3(
                    concat(
                         a1,
                         [
                            func2( (0)h21 , (0)h32 + (0)h42 ) +
                            func2( (0)h51 , 0               )
                         ]
                        )
                 )

    Similarly, h2, h3, h4 and h5 are calculated.

    Next, all atom hidden states are concatenated to make a feature vector of the molecule:

    ``mol_encodings = [[h1, h2, h3, h4, h5]]``

    **Concatenate the global features**

    Let,
    ``global_features = [[gf1, gf2, gf3]]``
    This array contains molecule level features. In case of this example, it contains 3 global features.

    Hence after concatenation,

    ``mol_encodings = [[h1, h2, h3, h4, h5, gf1, gf2, gf3]]``
    (Final output of the encoder)

    References
    ----------
    .. [1] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

    Examples
    --------
    >>> from rdkit import Chem
    >>> import torch
    >>> import deepchem as dc
    >>> input_smile = "CC"
    >>> feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
    >>> graph = feat.featurize(input_smile)
    >>> from deepchem.models.torch_models.dmpnn import _MapperDMPNN
    >>> mapper = _MapperDMPNN(graph[0])
    >>> atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values
    >>> atom_features = torch.from_numpy(atom_features).float()
    >>> f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
    >>> atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds)
    >>> mapping = torch.from_numpy(mapping)
    >>> global_features = torch.from_numpy(global_features).float()
    >>> molecules_unbatch_key = len(atom_features)
    >>> layer = DMPNNEncoderLayer(d_hidden=2)
    >>> output = layer(atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features, molecules_unbatch_key)
    """

    def __init__(self,
                 use_default_fdim: bool = True,
                 atom_fdim: int = 133,
                 bond_fdim: int = 14,
                 d_hidden: int = 300,
                 depth: int = 3,
                 bias: bool = False,
                 activation: str = 'relu',
                 dropout_p: float = 0.0,
                 aggregation: str = 'mean',
                 aggregation_norm: Union[int, float] = 100):
        """Initialize a DMPNNEncoderLayer layer.

        Parameters
        ----------
        use_default_fdim: bool
            If ``True``, ``self.atom_fdim`` and ``self.bond_fdim`` are initialized using values from the GraphConvConstants class. If ``False``, ``self.atom_fdim`` and ``self.bond_fdim`` are initialized from the values provided.
        atom_fdim: int
            Dimension of atom feature vector.
        bond_fdim: int
            Dimension of bond feature vector.
        d_hidden: int
            Size of hidden layer in the encoder layer.
        depth: int
            No of message passing steps.
        bias: bool
            If ``True``, dense layers will use bias vectors.
        activation: str
            Activation function to be used in the encoder layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
        dropout_p: float
            Dropout probability for the encoder layer.
        aggregation: str
            Aggregation type to be used in the encoder layer.
            Can choose between 'mean', 'sum', and 'norm'.
        aggregation_norm: Union[int, float]
            Value required if `aggregation` type is 'norm'.
        """
        super(DMPNNEncoderLayer, self).__init__()

        if use_default_fdim:
            from deepchem.feat.molecule_featurizers.dmpnn_featurizer import GraphConvConstants
            self.atom_fdim: int = GraphConvConstants.ATOM_FDIM
            self.concat_fdim: int = GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM
        else:
            self.atom_fdim = atom_fdim
            self.concat_fdim = atom_fdim + bond_fdim

        self.depth: int = depth
        self.aggregation: str = aggregation
        self.aggregation_norm: Union[int, float] = aggregation_norm

        if activation == 'relu':
            self.activation: nn.modules.activation.Module = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        self.dropout: nn.modules.dropout.Module = nn.Dropout(dropout_p)

        # Input
        self.W_i: nn.Linear = nn.Linear(self.concat_fdim, d_hidden, bias=bias)

        # Shared weight matrix across depths (default):
        # For messages hidden states
        self.W_h: nn.Linear = nn.Linear(d_hidden, d_hidden, bias=bias)

        # For atom hidden states
        self.W_o: nn.Linear = nn.Linear(self.atom_fdim + d_hidden, d_hidden)

    def _get_updated_atoms_hidden_state(
            self, atom_features: torch.Tensor, h_message: torch.Tensor,
            atom_to_incoming_bonds: torch.Tensor) -> torch.Tensor:
        """
        Method to compute atom hidden states.

        Parameters
        ----------
        atom_features: torch.Tensor
            Tensor containing atoms features.
        h_message: torch.Tensor
            Tensor containing hidden states of messages.
        atom_to_incoming_bonds: torch.Tensor
            Tensor containing mapping from atom index to list of indicies of incoming bonds.

        Returns
        -------
        atoms_hidden_states: torch.Tensor
            Tensor containing atom hidden states.
        """
        messages_to_atoms: torch.Tensor = h_message[atom_to_incoming_bonds].sum(
            1)  # num_atoms x hidden_size
        atoms_hidden_states: torch.Tensor = self.W_o(
            torch.cat((atom_features, messages_to_atoms),
                      1))  # num_atoms x hidden_size
        atoms_hidden_states = self.activation(
            atoms_hidden_states)  # num_atoms x hidden_size
        atoms_hidden_states = self.dropout(
            atoms_hidden_states)  # num_atoms x hidden_size
        return atoms_hidden_states  # num_atoms x hidden_size

    def _readout(self, atoms_hidden_states: torch.Tensor,
                 molecules_unbatch_key: List) -> torch.Tensor:
        """
        Method to execute the readout phase. (compute molecules encodings from atom hidden states)

        Parameters
        ----------
        atoms_hidden_states: torch.Tensor
            Tensor containing atom hidden states.
        molecules_unbatch_key: List
            List containing number of atoms in various molecules of a batch

        Returns
        -------
        molecule_hidden_state: torch.Tensor
            Tensor containing molecule encodings.
        """
        mol_vecs: List = []
        atoms_hidden_states_split: Sequence[Tensor] = torch.split(
            atoms_hidden_states, molecules_unbatch_key)
        mol_vec: torch.Tensor
        for mol_vec in atoms_hidden_states_split:
            if self.aggregation == 'mean':
                mol_vec = mol_vec.sum(dim=0) / len(mol_vec)
            elif self.aggregation == 'sum':
                mol_vec = mol_vec.sum(dim=0)
            elif self.aggregation == 'norm':
                mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
            else:
                raise Exception("Invalid aggregation")
            mol_vecs.append(mol_vec)

        molecule_hidden_state: torch.Tensor = torch.stack(mol_vecs, dim=0)
        return molecule_hidden_state  # num_molecules x hidden_size

    def forward(self, atom_features: torch.Tensor,
                f_ini_atoms_bonds: torch.Tensor,
                atom_to_incoming_bonds: torch.Tensor, mapping: torch.Tensor,
                global_features: torch.Tensor,
                molecules_unbatch_key: List) -> torch.Tensor:
        """
        Output computation for the DMPNNEncoderLayer.

        Steps:

        - Get original bond hidden states from concatenation of initial atom and bond features. (``input``)
        - Get initial messages hidden states. (``message``)
        - Execute message passing step for ``self.depth - 1`` iterations.
        - Get atom hidden states using atom features and message hidden states.
        - Get molecule encodings.
        - Concatenate global molecular features and molecule encodings.

        Parameters
        ----------
        atom_features: torch.Tensor
            Tensor containing atoms features.
        f_ini_atoms_bonds: torch.Tensor
            Tensor containing concatenated feature vector which contains concatenation of initial atom and bond features.
        atom_to_incoming_bonds: torch.Tensor
            Tensor containing mapping from atom index to list of indicies of incoming bonds.
        mapping: torch.Tensor
            Tensor containing the mapping that maps bond index to 'array of indices of the bonds'
            incoming at the initial atom of the bond (excluding the reverse bonds).
        global_features: torch.Tensor
            Tensor containing molecule features.
        molecules_unbatch_key: List
            List containing number of atoms in various molecules of a batch

        Returns
        -------
        output: torch.Tensor
            Tensor containing the encodings of the molecules.
        """
        input: torch.Tensor = self.W_i(
            f_ini_atoms_bonds)  # num_bonds x hidden_size
        message: torch.Tensor = self.activation(
            input)  # num_bonds x hidden_size

        for _ in range(1, self.depth):
            message = message[mapping].sum(1)  # num_bonds x hidden_size
            h_message: torch.Tensor = input + self.W_h(
                message)  # num_bonds x hidden_size
            h_message = self.activation(h_message)  # num_bonds x hidden_size
            h_message = self.dropout(h_message)  # num_bonds x hidden_size

        # num_atoms x hidden_size
        atoms_hidden_states: torch.Tensor = self._get_updated_atoms_hidden_state(
            atom_features, h_message, atom_to_incoming_bonds)

        # num_molecules x hidden_size
        output: torch.Tensor = self._readout(atoms_hidden_states,
                                             molecules_unbatch_key)

        # concat global features
        if global_features.size()[0] != 0:
            if len(global_features.shape) == 1:
                global_features = global_features.view(len(output), -1)
            output = torch.cat([output, global_features], dim=1)

        return output  # num_molecules x hidden_size


class InteratomicL2Distances(nn.Module):
    """Compute (squared) L2 Distances between atoms given neighbors.

    This class is the pytorch implementation of the original InteratomicL2Distances layer implemented in Keras.
    Pairwise distance (L2) is calculated between input atoms, given the number of neighbors to consider, along with the number of descriptors for every atom.

    Examples
    --------
    >>> atoms = 5
    >>> neighbors = 2
    >>> coords = np.random.rand(atoms, 3)
    >>> neighbor_list = np.random.randint(0, atoms, size=(atoms, neighbors))
    >>> layer = InteratomicL2Distances(atoms, neighbors, 3)
    >>> result = np.array(layer([coords, neighbor_list]))
    >>> result.shape
    (5, 2)

    """

    def __init__(self, N_atoms: int, M_nbrs: int, ndim: int, **kwargs):
        """Constructor for this layer.

        Parameters
        ----------
        N_atoms: int
            Number of atoms in the system total.
        M_nbrs: int
            Number of neighbors to consider when computing distances.
        n_dim:  int
            Number of descriptors for each atom.
        """
        super(InteratomicL2Distances, self).__init__(**kwargs)
        self.N_atoms = N_atoms
        self.M_nbrs = M_nbrs
        self.ndim = ndim

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(N_atoms={self.N_atoms}, M_nbrs={self.M_nbrs}, ndim={self.ndim})'
        )

    def forward(
        self, inputs: List[Union[torch.Tensor,
                                 List[Union[int, float]]]]) -> torch.Tensor:
        """Invokes this layer.

        Parameters
        ----------
        inputs: list
            Should be of form `inputs=[coords, nbr_list]` where `coords` is a
            tensor of shape `(None, N, 3)` and `nbr_list` is a list.

        Returns
        -------
        Tensor of shape `(N_atoms, M_nbrs)` with interatomic distances.
        """
        if len(inputs) != 2:
            raise ValueError("InteratomicDistances requires coords,nbr_list")
        coords, nbr_list = (torch.tensor(inputs[0]), torch.tensor(inputs[1]))
        N_atoms, M_nbrs, ndim = self.N_atoms, self.M_nbrs, self.ndim
        # Shape (N_atoms, M_nbrs, ndim)
        nbr_coords = coords[nbr_list.long()]
        # Shape (N_atoms, M_nbrs, ndim)
        tiled_coords = torch.tile(torch.reshape(coords, (N_atoms, 1, ndim)),
                                  (1, M_nbrs, 1))
        # Shape (N_atoms, M_nbrs)
        return torch.sum((tiled_coords - nbr_coords)**2, dim=2)


class RealNVPLayer(nn.Module):
    """Real NVP Transformation Layer

    This class class is a constructor transformation layer used on a
    NormalizingFLow model.  The Real Non-Preserving-Volumen (Real NVP) is a type
    of normalizing flow layer which gives advantages over this mainly because an
    ease to compute the inverse pass [1]_, this is to learn a target
    distribution.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.distributions import MultivariateNormal
    >>> from deepchem.models.torch_models.layers import RealNVPLayer
    >>> dim = 2
    >>> samples = 96
    >>> data = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> tensor = data.sample(torch.Size((samples, dim)))

    >>> layers = 4
    >>> hidden_size = 16
    >>> masks = F.one_hot(torch.tensor([i % 2 for i in range(layers)])).float()
    >>> layers = nn.ModuleList([RealNVPLayer(mask, hidden_size) for mask in masks])

    >>> for layer in layers:
    ...   _, inverse_log_det_jacobian = layer.inverse(tensor)
    ...   inverse_log_det_jacobian = inverse_log_det_jacobian.detach().numpy()
    >>> len(inverse_log_det_jacobian)
    96

    References
    ----------
    .. [1] Stimper, V., Schölkopf, B., & Hernández-Lobato, J. M. (2021). Resampling Base
    Distributions of Normalizing Flows. (2017). Retrieved from http://arxiv.org/abs/2110.15828
    """

    def __init__(self, mask: torch.Tensor, hidden_size: int) -> None:
        """
        Parameters
        -----------
        mask: torch.Tensor
            Tensor with zeros and ones and its size depende on the number of layers
            and dimenssions the user request.
        hidden_size: int
            The size of the outputs and inputs used on the internal nodes of the
            transformation layer.

        """
        super(RealNVPLayer, self).__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.dim = len(mask)

        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim))

        self.scale = nn.Parameter(torch.Tensor(self.dim))

        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim))

    def forward(self, x: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        This particular transformation is represented by the following function:
        y = x + (1 - x) * exp( s(x)) + t(x), where t and s needs an activation
        function. This class also returns the logarithm of the jacobians
        determinant which is useful when invert a transformation and compute
        the probability of the transformation.

        Parameters
        ----------
        x : Sequence
            Tensor sample with the initial distribution data which will pass into
            the normalizing algorithm

        Returns
        -------
        y : torch.Tensor
            Transformed tensor according to Real NVP layer with the shape of 'x'.
        log_det_jacobian : torch.Tensor
            Tensor which represents the info about the deviation of the initial
            and target distribution.

        """
        x_mask = x * self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)

        log_det_jacobian = ((1 - self.mask) * s).sum(-1)
        return y, log_det_jacobian

    def inverse(self, y: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Inverse pass

        This class performs the inverse of the previous method (formward).
        Also, this metehod returns the logarithm of the jacobians determinant
        which is useful to compute the learneable features of target distribution.

        Parameters
        ----------
        y : Sequence
            Tensor sample with transformed distribution data which will be used in
            the normalizing algorithm inverse pass.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor according to Real NVP layer with the shape of 'y'.
        inverse_log_det_jacobian : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.

        """
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1 - self.mask) * (y - t) * torch.exp(-s)

        inverse_log_det_jacobian = ((1 - self.mask) * -s).sum(-1)

        return x, inverse_log_det_jacobian


class NeighborList(nn.Module):
    """Computes a neighbor-list in PyTorch.

    Neighbor-lists (also called Verlet Lists) are a tool for grouping
    atoms which are close to each other spatially. This layer computes a
    Neighbor List from a provided tensor of atomic coordinates. You can
    think of this as a general "k-means" layer, but optimized for the
    case `k==3`.

    Examples
    --------
    >>> N_atoms = 5
    >>> start = 0
    >>> stop = 12
    >>> nbr_cutoff = 3
    >>> ndim = 3
    >>> M_nbrs = 2
    >>> coords = start + np.random.rand(N_atoms, ndim) * (stop - start)
    >>> coords = torch.as_tensor(coords, dtype=torch.float)
    >>> layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
    ...                      stop)
    >>> result = layer(coords)
    >>> result.shape
    torch.Size([5, 2])

    TODO(rbharath): Make this layer support batching.
    """

    def __init__(self, N_atoms: int, M_nbrs: int, ndim: int,
                 nbr_cutoff: Union[int,
                                   float], start: int, stop: int, **kwargs):
        """
        Parameters
        ----------
        N_atoms: int
            Maximum number of atoms this layer will neighbor-list.
        M_nbrs: int
            Maximum number of spatial neighbors possible for atom.
        ndim: int
            Dimensionality of space atoms live in. (Typically 3D, but sometimes will
            want to use higher dimensional descriptors for atoms).
        nbr_cutoff: int or float
            Length in Angstroms (?) at which atom boxes are gridded.
        start: int
            Start of range for the box in which the locations of all grid points will be calculated in `self.get_cells()`.
        stop: int
            End of range for the box in which the locations of all grid points will be calculated in `self.get_cells()`.
        """
        super(NeighborList, self).__init__(**kwargs)
        self.N_atoms = N_atoms
        self.M_nbrs = M_nbrs
        self.ndim = ndim
        # Number of grid cells
        n_cells = int(((stop - start) / nbr_cutoff)**ndim)
        self.n_cells = n_cells
        self.nbr_cutoff = nbr_cutoff
        self.start = start
        self.stop = stop

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(N_atoms={self.N_atoms}, M_nbrs={self.M_nbrs}, ndim={self.ndim}, n_cells={self.n_cells}, nbr_cutoff={self.nbr_cutoff}, start={self.start}, stop={self.stop})'
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Invokes this layer.

        Parameters
        ----------
        inputs: torch.Tensor
            Shape (num_atoms, ndim)

        Returns
        -------
        neighbor_list: torch.Tensor
            Shape `(N_atoms, M_nbrs)`
        """
        if isinstance(inputs, SequenceCollection):
            if len(inputs) != 1:
                raise ValueError("NeighborList can only have one input")
            inputs = inputs[0]
        if len(inputs.shape) != 2:
            # TODO(rbharath): Support batching
            raise ValueError("Parent tensor must be (num_atoms, ndum)")
        return self.compute_nbr_list(inputs)

    def compute_nbr_list(self, coords: torch.Tensor) -> torch.Tensor:
        """Get closest neighbors for atoms.

        Needs to handle padding for atoms with no neighbors.

        Parameters
        ----------
        coords: torch.Tensor
            Shape (N_atoms, ndim)

        Returns
        -------
        nbr_list: torch.Tensor
            Shape (N_atoms, M_nbrs) of atom indices
        """
        # Shape (n_cells, ndim)
        cells = self.get_cells()

        # List of length N_atoms, each element of different length uniques_i
        nbrs = self.get_atoms_in_nbrs(coords, cells)
        padding = torch.full((self.M_nbrs,), -1)
        padded_nbrs = [
            torch.concat([unique_nbrs, padding], 0) for unique_nbrs in nbrs
        ]

        # List of length N_atoms, each element of different length uniques_i
        # List of length N_atoms, each a tensor of shape
        # (uniques_i, ndim)
        nbr_coords = [
            torch.index_select(coords, 0, atom_nbrs) for atom_nbrs in nbrs
        ]

        # Add phantom atoms that exist far outside the box
        coord_padding = torch.full((self.M_nbrs, self.ndim),
                                   2 * self.stop).to(torch.float)
        padded_nbr_coords = [
            torch.cat([nbr_coord, coord_padding], 0) for nbr_coord in nbr_coords
        ]

        # List of length N_atoms, each of shape (1, ndim)
        atom_coords = torch.tensor_split(coords, self.N_atoms)

        # TODO(rbharath): How does distance need to be modified here to
        # account for periodic boundary conditions?
        # List of length N_atoms each of shape (M_nbrs)
        padded_dists = [
            torch.sum((atom_coord - padded_nbr_coord)**2, dim=-1)
            for (atom_coord,
                 padded_nbr_coord) in zip(atom_coords, padded_nbr_coords)
        ]

        padded_closest_nbrs = [
            torch.topk(padded_dist, k=self.M_nbrs, largest=False)[1]
            for padded_dist in padded_dists
        ]

        # N_atoms elts of size (M_nbrs,) each
        padded_neighbor_list = [
            torch.gather(padded_atom_nbrs, 0, padded_closest_nbr)
            for (padded_atom_nbrs,
                 padded_closest_nbr) in zip(padded_nbrs, padded_closest_nbrs)
        ]

        neighbor_list = torch.stack(padded_neighbor_list)

        return neighbor_list

    def get_atoms_in_nbrs(self, coords: torch.Tensor,
                          cells: torch.Tensor) -> List[torch.Tensor]:
        """Get the atoms in neighboring cells for each cells.

        Parameters
        ----------
        coords: torch.Tensor
            Shape (N_atoms, ndim)
        cells: torch.Tensor

        Returns
        -------
        atoms_in_nbrs: List[torch.Tensor]
            (N_atoms, n_nbr_cells, M_nbrs)
        """
        # Shape (N_atoms, 1)
        cells_for_atoms = self.get_cells_for_atoms(coords, cells)

        # Find M_nbrs atoms closest to each cell
        # Shape (n_cells, M_nbrs)
        closest_atoms = self.get_closest_atoms(coords, cells)

        # Associate each cell with its neighbor cells. Assumes periodic boundary
        # conditions, so does wrapround. O(constant)
        # Shape (n_cells, n_nbr_cells)
        neighbor_cells = self.get_neighbor_cells(cells)

        # Shape (N_atoms, n_nbr_cells)
        neighbor_cells = torch.squeeze(
            torch.index_select(neighbor_cells, 0,
                               torch.squeeze(cells_for_atoms)))

        # Shape (N_atoms, n_nbr_cells, M_nbrs)
        atoms_in_nbrs = torch.index_select(closest_atoms, 0,
                                           neighbor_cells.flatten())

        # Shape (N_atoms, n_nbr_cells*M_nbrs)
        atoms_in_nbrs = torch.reshape(atoms_in_nbrs, [self.N_atoms, -1])

        # List of length N_atoms, each element length uniques_i
        nbrs_per_atom = torch.split(atoms_in_nbrs, self.N_atoms)

        uniques = [
            torch.unique(atom_nbrs, sorted=False)
            for atom_nbrs in nbrs_per_atom[0]
        ]

        # TODO(rbharath): FRAGILE! Uses fact that identity seems to be the first
        # element removed to remove self from list of neighbors. Need to verify
        # this holds more broadly or come up with robust alternative.
        uniques = [unique[1:] for unique in uniques]

        return uniques

    def get_closest_atoms(self, coords: torch.Tensor,
                          cells: torch.Tensor) -> torch.Tensor:
        """For each cell, find M_nbrs closest atoms.

        Let N_atoms be the number of atoms.

        Parameters
        ----------
        coords: torch.Tensor
            (N_atoms, ndim) shape.
        cells: torch.Tensor
            (n_cells, ndim) shape.

        Returns
        -------
        closest_inds: torch.Tensor
            Of shape (n_cells, M_nbrs)
        """
        N_atoms, n_cells, ndim, M_nbrs = (self.N_atoms, self.n_cells, self.ndim,
                                          self.M_nbrs)
        # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
        tiled_cells = torch.reshape(torch.tile(cells, (1, N_atoms)),
                                    (N_atoms * n_cells, ndim))

        # Shape (N_atoms*n_cells, ndim) after tile
        tiled_coords = torch.tile(coords, (n_cells, 1))

        # Shape (N_atoms*n_cells)
        coords_vec = torch.sum((tiled_coords - tiled_cells)**2, dim=-1)
        # Shape (n_cells, N_atoms)
        coords_norm = torch.reshape(coords_vec, (n_cells, N_atoms))

        # Find k atoms closest to this cell.
        # Tensor of shape (n_cells, M_nbrs)
        closest_inds = torch.topk(coords_norm, k=M_nbrs, largest=False)[1]

        return closest_inds

    def get_cells_for_atoms(self, coords: torch.Tensor,
                            cells: torch.Tensor) -> torch.Tensor:
        """Compute the cells each atom belongs to.

        Parameters
        ----------
        coords: torch.Tensor
            Shape (N_atoms, ndim)
        cells: torch.Tensor
            (n_cells, ndim) shape.
        Returns
        -------
        cells_for_atoms: torch.Tensor
            Shape (N_atoms, 1)
        """
        N_atoms, n_cells, ndim = self.N_atoms, self.n_cells, self.ndim
        n_cells = int(n_cells)
        # Tile both cells and coords to form arrays of size (N_atoms*n_cells, ndim)
        tiled_cells = torch.tile(cells, (N_atoms, 1))

        # Shape (N_atoms*n_cells, 1) after tile
        tiled_coords = torch.reshape(torch.tile(coords, (1, n_cells)),
                                     (n_cells * N_atoms, ndim))
        coords_vec = torch.sum((tiled_coords - tiled_cells)**2, dim=-1)
        coords_norm = torch.reshape(coords_vec, (N_atoms, n_cells))

        closest_inds = torch.topk(coords_norm, k=1, largest=False)[1]

        return closest_inds

    def _get_num_nbrs(self) -> int:
        """Get number of neighbors in current dimensionality space."""
        return 3**self.ndim

    def get_neighbor_cells(self, cells: torch.Tensor) -> torch.Tensor:
        """Compute neighbors of cells in grid.

        # TODO(rbharath): Do we need to handle periodic boundary conditions
        properly here?
        # TODO(rbharath): This doesn't handle boundaries well. We hard-code
        # looking for n_nbr_cells neighbors, which isn't right for boundary cells in
        # the cube.

        Parameters
        ----------
        cells: torch.Tensor
            (n_cells, ndim) shape.
        Returns
        -------
        nbr_cells: torch.Tensor
            (n_cells, n_nbr_cells)
        """
        ndim, n_cells = self.ndim, self.n_cells
        n_nbr_cells = self._get_num_nbrs()
        # Tile cells to form arrays of size (n_cells*n_cells, ndim)
        # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
        # Tile (a, a, a, b, b, b, etc.)
        tiled_centers = torch.reshape(torch.tile(cells, (1, n_cells)),
                                      (n_cells * n_cells, ndim))
        # Tile (a, b, c, a, b, c, ...)
        tiled_cells = torch.tile(cells, (n_cells, 1))

        coords_vec = torch.sum((tiled_centers - tiled_cells)**2, dim=-1)
        coords_norm = torch.reshape(coords_vec, (n_cells, n_cells))
        closest_inds = torch.topk(coords_norm, k=n_nbr_cells, largest=False)[1]

        return closest_inds

    def get_cells(self) -> torch.Tensor:
        """Returns the locations of all grid points in box.

        Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
        Then would return a list of length 20^3 whose entries would be
        [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

        Returns
        -------
        cells: torch.Tensor
          (n_cells, ndim) shape.
        """
        start, stop, nbr_cutoff = self.start, self.stop, self.nbr_cutoff
        mesh_args = [
            torch.arange(start, stop, nbr_cutoff) for _ in range(self.ndim)
        ]
        return torch.reshape(
            torch.permute(
                torch.stack(torch.meshgrid(*mesh_args, indexing='xy')),
                tuple(range(self.ndim, -1, -1))),
            (self.n_cells, self.ndim)).to(torch.float)


class LSTMStep(nn.Module):
    """Layer that performs a single step LSTM update.

    This is the Torch equivalent of the original implementation using Keras.
    """

    def __init__(self,
                 output_dim,
                 input_dim,
                 init_fn='xavier_uniform_',
                 inner_init_fn='orthogonal_',
                 activation_fn='tanh',
                 inner_activation_fn='hardsigmoid',
                 **kwargs):
        """
        Parameters
        ----------
        output_dim: int
            Dimensionality of output vectors.
        input_dim: int
            Dimensionality of input vectors.
        init_fn: str
            PyTorch initialization to use for W.
        inner_init_fn: str
            PyTorch initialization to use for U.
        activation_fn: str
            PyTorch activation to use for output.
        inner_activation_fn: str
            PyTorch activation to use for inner steps.
        """

        super(LSTMStep, self).__init__(**kwargs)
        self.init = init_fn
        self.inner_init = inner_init_fn
        self.output_dim = output_dim
        # No other forget biases supported right now.
        self.activation = activation_fn
        self.inner_activation = inner_activation_fn
        self.activation_fn = get_activation(activation_fn)
        self.inner_activation_fn = get_activation(inner_activation_fn)
        self.input_dim = input_dim
        self.build()

    def get_config(self):
        config = super(LSTMStep, self).get_config()
        config['output_dim'] = self.output_dim
        config['input_dim'] = self.input_dim
        config['init_fn'] = self.init
        config['inner_init_fn'] = self.inner_init
        config['activation_fn'] = self.activation
        config['inner_activation_fn'] = self.inner_activation
        return config

    def get_initial_states(self, input_shape):
        return [torch.zeros(input_shape), torch.zeros(input_shape)]

    def build(self):
        """Constructs learnable weights for this layer."""
        init = getattr(initializers, self.init)
        inner_init = getattr(initializers, self.inner_init)
        self.W = init(torch.empty(self.input_dim, 4 * self.output_dim))
        self.U = inner_init(torch.empty(self.output_dim, 4 * self.output_dim))

        self.b = torch.tensor(np.hstack(
            (np.zeros(self.output_dim), np.ones(self.output_dim),
             np.zeros(self.output_dim), np.zeros(self.output_dim))),
                              dtype=torch.float32)

    def forward(self, inputs):
        """Execute this layer on input tensors.

        Parameters
        ----------
        inputs: list
            List of three tensors (x, h_tm1, c_tm1). h_tm1 means "h, t-1".

        Returns
        -------
        list
            Returns h, [h, c]
        """
        x, h_tm1, c_tm1 = inputs
        x, h_tm1, c_tm1 = torch.tensor(x), torch.tensor(h_tm1), torch.tensor(
            c_tm1)

        z = torch.matmul(x, self.W) + torch.matmul(h_tm1, self.U) + self.b

        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim:2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim:3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = self.inner_activation_fn(z0)
        f = self.inner_activation_fn(z1)
        c = f * c_tm1 + i * self.activation_fn(z2)
        o = self.inner_activation_fn(z3)

        h = o * self.activation_fn(c)
        return h, [h, c]


class AtomicConvolution(nn.Module):
    """Implements the Atomic Convolutional transform, introduced in

    Gomes, Joseph, et al. "Atomic convolutional networks for predicting
    protein-ligand binding affinity." arXiv preprint arXiv:1703.10603
    (2017).

    At a high level, this transform performs a graph convolution
    on the nearest neighbors graph in 3D space.

    Examples
    --------
    >>> batch_size = 4
    >>> max_atoms = 5
    >>> max_neighbors = 2
    >>> dimensions = 3
    >>> radial_params = torch.tensor([[5.0, 2.0, 0.5], [10.0, 2.0, 0.5],
    ...                               [5.0, 1.0, 0.2]])
    >>> input1 = np.random.rand(batch_size, max_atoms, dimensions).astype(np.float32)
    >>> input2 = np.random.randint(max_atoms,
    ...                            size=(batch_size, max_atoms, max_neighbors))
    >>> input3 = np.random.randint(1, 10, size=(batch_size, max_atoms, max_neighbors))
    >>> layer = AtomicConvolution(radial_params=radial_params)
    >>> result = layer([input1, input2, input3])
    >>> result.shape
    torch.Size([4, 5, 3])
    """

    def __init__(self,
                 atom_types: Optional[Union[ArrayLike, torch.Tensor]] = None,
                 radial_params: Union[ArrayLike, torch.Tensor] = list(),
                 box_size: Optional[Union[ArrayLike, torch.Tensor]] = None,
                 **kwargs):
        """Initialize this layer.

        Parameters
        ----------
        atom_types : Union[ArrayLike, torch.Tensor], optional
            List of atom types.
        radial_params : Union[ArrayLike, torch.Tensor], optional
            List of radial params.
        box_size : Union[ArrayLike, torch.Tensor], optional
            Length must be equal to the number of features.
        """

        super(AtomicConvolution, self).__init__(**kwargs)
        self.atom_types = atom_types
        self.radial_params = radial_params

        if box_size is None or isinstance(box_size, torch.Tensor):
            self.box_size = box_size
        else:
            self.box_size = torch.tensor(box_size)

        vars = []
        for i in range(3):
            val = np.array([p[i] for p in self.radial_params]).reshape(
                (-1, 1, 1, 1))
            vars.append(torch.tensor(val, dtype=torch.float))
        self.rc = nn.Parameter(vars[0])
        self.rs = nn.Parameter(vars[1])
        self.re = nn.Parameter(vars[2])

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(atom_types={self.atom_types}, radial_params={self.radial_params}, box_size={self.box_size}, rc={self.rc}, rs={self.rs}, re={self.re})'
        )

    def forward(
            self, inputs: Sequence[Union[ArrayLike,
                                         torch.Tensor]]) -> torch.Tensor:
        """Invoke this layer.

        B, N, M, d, l = batch_size, max_num_atoms, max_num_neighbors, num_features, len(radial_params) * len(atom_types)

        Parameters
        ----------
        inputs: Sequence[Union[ArrayLike, torch.Tensor]]
            First input are the coordinates/features, of shape (B, N, d)
            Second input is the neighbor list, of shape (B, N, M)
            Third input are the atomic numbers of neighbor atoms, of shape (B, N, M)

        Returns
        -------
        torch.Tensor of shape (B, N, l)
            Output of atomic convolution layer.

        Raises
        ------
        ValueError
            When the length of `inputs` is not equal to 3.
        """
        if len(inputs) != 3:
            raise ValueError(
                f"`inputs` has to be of length 3, got: {len(inputs)}")

        X = torch.tensor(inputs[0])
        Nbrs = torch.tensor(inputs[1], dtype=torch.int64)
        Nbrs_Z = torch.tensor(inputs[2])

        B, N, d = X.shape
        M = Nbrs.shape[-1]

        D = self.distance_tensor(X, Nbrs, self.box_size, B, N, M, d)
        R = self.distance_matrix(D)
        R = torch.unsqueeze(R, 0)
        rsf = self.radial_symmetry_function(R, self.rc, self.rs, self.re)

        if not self.atom_types:
            cond = torch.not_equal(Nbrs_Z, 0).to(torch.float).reshape(
                (1, -1, N, M))
            layer = torch.sum(cond * rsf, 3)
        else:
            # Sum the pairwise-interactions between atoms that are of `atom_type` and its neighbors for each atom type in `atom_types`.
            symmetries = []
            for atom_type in self.atom_types:
                cond = torch.eq(Nbrs_Z, atom_type).to(torch.float).reshape(
                    (1, -1, N, M))
                symmetries.append(torch.sum(cond * rsf, 3))
            layer = torch.concat(symmetries, 0)

        layer = torch.permute(layer, [1, 2, 0])
        var, mean = torch.var_mean(layer, [0, 2])
        var, mean = var.detach(), mean.detach()

        return F.batch_norm(layer, mean, var)

    def distance_tensor(self, X: torch.Tensor, Nbrs: torch.Tensor,
                        box_size: Union[torch.Tensor, None], B: int, N: int,
                        M: int, d: int) -> torch.Tensor:
        """Calculate distance tensor for a batch of molecules.

        B, N, M, d = batch_size, max_num_atoms, max_num_neighbors, num_features

        Parameters
        ----------
        X : torch.Tensor of shape (B, N, d)
            Coordinates/features.
        Nbrs : torch.Tensor of shape (B, N, M)
            Neighbor list.
        box_size : torch.Tensor
            Length must be equal to `d`.
        B : int
            Batch size
        N : int
            Maximum number of atoms
        M : int
            Maximum number of neighbors
        d : int
            Number of features

        Returns
        -------
        torch.Tensor of shape (B, N, M, d)
            Coordinates/features distance tensor.

        Raises
        ------
        ValueError
            When the length of `box_size` is not equal to `d`.
        """
        if box_size is not None and len(box_size) != d:
            raise ValueError("Length of `box_size` must be equal to `d`")

        flat_neighbors = torch.reshape(Nbrs, (-1, N * M))
        neighbor_coords = torch.stack(
            [X[b, flat_neighbors[b]] for b in range(B)])
        neighbor_coords = torch.reshape(neighbor_coords, (-1, N, M, d))
        D = neighbor_coords - torch.unsqueeze(X, 2)
        if box_size is not None:
            box_size = torch.reshape(box_size, (1, 1, 1, d))
            D -= torch.round(D / box_size) * box_size

        return D

    def distance_matrix(self, D: torch.Tensor) -> torch.Tensor:
        """Calculate a distance matrix, given a distance tensor.

        B, N, M, d = batch_size, max_num_atoms, max_num_neighbors, num_features

        Parameters
        ----------
        D : torch.Tensor of shape (B, N, M, d)
            Distance tensor

        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Distance matrix.
        """
        return torch.sqrt(torch.sum(torch.mul(D, D), 3))

    def gaussian_distance_matrix(self, R: torch.Tensor, rs: torch.Tensor,
                                 re: torch.Tensor) -> torch.Tensor:
        """Calculate a Gaussian distance matrix.

        B, N, M, l = batch_size, max_num_atoms, max_num_neighbors, len(radial_params)

        Parameters
        ----------
        R : torch.Tensor of shape (B, N, M)
            Distance matrix.
        rs : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix mean.
        re : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix width.

        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Gaussian distance matrix.
        """
        return torch.exp(-re * (R - rs)**2)

    def radial_cutoff(self, R: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        """Calculate a radial cut-off matrix.

        B, N, M, l = batch_size, max_num_atoms, max_num_neighbors, len(radial_params)

        Parameters
        ----------
        R : torch.Tensor of shape (B, N, M)
            Distance matrix.
        rc : torch.Tensor of shape (l, 1, 1, 1)
            Interaction cutoff (in angstrom).

        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Radial cutoff matrix.
        """
        T = 0.5 * (torch.cos(np.pi * R / rc) + 1)
        E = torch.zeros_like(T)
        cond = torch.less_equal(R, rc)
        FC = torch.where(cond, T, E)

        return FC

    def radial_symmetry_function(self, R: torch.Tensor, rc: torch.Tensor,
                                 rs: torch.Tensor,
                                 re: torch.Tensor) -> torch.Tensor:
        """Calculate a radial symmetry function.

        B, N, M, l = batch_size, max_num_atoms, max_num_neighbors, len(radial_params)

        Parameters
        ----------
        R : torch.Tensor of shape (B, N, M)
            Distance matrix.
        rc : torch.Tensor of shape (l, 1, 1, 1)
            Interaction cutoff (in angstrom).
        rs : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix mean.
        re : torch.Tensor of shape (l, 1, 1, 1)
            Gaussian distance matrix width.
        Returns
        -------
        torch.Tensor of shape (B, N, M)
            Pre-summation radial symmetry function.
        """
        K = self.gaussian_distance_matrix(R, rs, re)
        FC = self.radial_cutoff(R, rc)

        return torch.mul(K, FC)


class CombineMeanStd(nn.Module):
    """Generate Gaussian noise.

    This is the Torch equivalent of the original implementation using Keras.
    """

    def __init__(self,
                 training_only: bool = False,
                 noise_epsilon: float = 1.0,
                 **kwargs):
        """Create a CombineMeanStd layer.

        This layer should have two inputs with the same shape, and its
        output also has the same shape.  Each element of the output is a
        Gaussian distributed random number whose mean is the corresponding
        element of the first input, and whose standard deviation is the
        corresponding element of the second input.

        Parameters
        ----------
        training_only: bool, optional (default False).
            if True, noise is only generated during training.  During
            prediction, the output is simply equal to the first input (that
            is, the mean of the distribution used during training).
        noise_epsilon: float, optional (default 1.0).
            The noise is scaled by this factor
        """
        super(CombineMeanStd, self).__init__(**kwargs)
        self.training_only = training_only
        self.noise_epsilon = noise_epsilon

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(training_only={self.training_only}, noise_epsilon={self.noise_epsilon})'
        )

    def forward(self,
                inputs: Sequence[ArrayLike],
                training: bool = True) -> torch.Tensor:
        """Invoke this layer.

        Parameters
        ----------
        inputs: Sequence[ArrayLike]
            First element are the means for the random generated numbers.
            Second element are the standard deviations for the random generated numbers.
        training: bool, optional (default True).
            Specifies whether to generate noise.
            Noise is only added when training.

        Returns
        -------
        Tensor of Gaussian distributed random numbers: torch.Tensor
            Same shape as the means and standard deviations from `inputs`.
        """
        if len(inputs) != 2:
            raise ValueError("Must have two in_layers")

        mean_parent, std_parent = torch.tensor(inputs[0]), torch.tensor(
            inputs[1])
        noise_scale = torch.tensor(training or
                                   not self.training_only).to(torch.float)
        sample_noise = torch.normal(0.0, self.noise_epsilon, mean_parent.shape)
        return mean_parent + noise_scale * std_parent * sample_noise


class GatedRecurrentUnit(nn.Module):
    """ Submodule for Message Passing """

    def __init__(self, n_hidden=100, init='xavier_uniform_', **kwargs):
        super(GatedRecurrentUnit, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.init = init
        init = getattr(initializers, self.init)
        self.Wz = init(torch.empty(n_hidden, n_hidden))
        self.Wr = init(torch.empty(n_hidden, n_hidden))
        self.Wh = init(torch.empty(n_hidden, n_hidden))
        self.Uz = init(torch.empty(n_hidden, n_hidden))
        self.Ur = init(torch.empty(n_hidden, n_hidden))
        self.Uh = init(torch.empty(n_hidden, n_hidden))
        self.bz = torch.zeros((n_hidden,))
        self.br = torch.zeros((n_hidden,))
        self.bh = torch.zeros((n_hidden,))

    def forward(self, inputs):
        sigmoid = get_activation('sigmoid')
        tanh = get_activation('tanh')
        h_tm1, x = inputs
        z = sigmoid(
            torch.matmul(x, self.Wz) + torch.matmul(h_tm1, self.Uz) + self.bz)
        r = sigmoid(
            torch.matmul(x, self.Wr) + torch.matmul(h_tm1, self.Ur) + self.br)
        h = (1 - z) * tanh(
            torch.matmul(x, self.Wh) + torch.matmul(h_tm1 * r, self.Uh) +
            self.bh) + z * x
        return h


class WeightedLinearCombo(nn.Module):
    """Compute a weighted linear combination of input layers, where the weight variables are trained.

    Examples
    --------
    >>> input1 = np.random.rand(5, 10).astype(np.float32)
    >>> input2 = np.random.rand(5, 10).astype(np.float32)
    >>> layer = WeightedLinearCombo(len([input1, input2]))
    >>> result = layer([input1, input2])
    >>> result.shape
    torch.Size([5, 10])
    """

    def __init__(self, num_inputs: int, std: float = 0.3, **kwargs):
        """

        Parameters
        ----------
        num_inputs: int
            Number of inputs given to `self.forward()`
            This is used to initialize the correct amount of weight variables to be trained.
        std: float
            The standard deviation for the normal distribution that is used to initialize the trainable weights.
        """
        super(WeightedLinearCombo, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.std = std
        self.input_weights = nn.Parameter(torch.empty(self.num_inputs))
        nn.init.normal_(self.input_weights, std=std)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(num_inputs={self.num_inputs}, std={self.std}, input_weights={self.input_weights})'
        )

    def forward(
        self, inputs: Sequence[Union[ArrayLike,
                                     torch.Tensor]]) -> Optional[torch.Tensor]:
        """
        Parameters
        ----------
        inputs: Sequence[Union[ArrayLike, torch.Tensor]]
            The initial input layers.
            The length must be the same as `self.num_inputs`.

        Returns
        -------
        out_tensor: torch.Tensor or None
            The tensor containing the weighted linear combination.
        """
        out_tensor = None
        for in_tensor, w in zip(inputs, self.input_weights):
            in_tensor = torch.FloatTensor(in_tensor)
            if out_tensor is None:
                out_tensor = w * in_tensor
            else:
                out_tensor += w * in_tensor
        return out_tensor
