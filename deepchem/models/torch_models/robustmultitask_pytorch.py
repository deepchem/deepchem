import logging
from collections.abc import Sequence as SequenceCollection
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')


from deepchem.utils.pytorch_utils import get_activation
logger = logging.getLogger(__name__)


class RobustMultitaskClassifier(nn.Module):
    """Implements a neural network for robust multitasking.

    The key idea of this model is to have bypass layers that feed
    directly from features to task output. This might provide some
    flexibility toroute around challenges in multitasking with
    destructive interference.

    References
    ----------
    This technique was introduced in [1]_

    .. [1] Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

    """
    def __init__(self,
                 n_tasks,
                 n_features,
                 layer_sizes=[1000,],
                 weight_init_stddevs=0.02,
                 bias_init_consts=1.0,
                 weight_decay_penalty=0.0,
                 weight_decay_penalty_type="l2",
                 dropouts=0.5,
                 activation_fns="relu",
                 n_classes=2,
                 bypass_layer_sizes=[100,],
                 bypass_weight_init_stddevs=[.02],
                 bypass_bias_init_consts=[1.],
                 bypass_dropouts=[.5],
                 ):
        """  Create a RobustMultitaskClassifier.

        Parameters
        ----------
        n_tasks: int
            number of tasks
        n_features: int
            number of features
        layer_sizes: list
            the size of each dense layer in the network.  The length of this list determines the number of layers.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization of each layer.  The length
            of this list should equal len(layer_sizes).  Alternatively this may be a single value instead of a list,
            in which case the same value is used for every layer.
        bias_init_consts: list or loat
            the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        weight_decay_penalty: float
            the magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str
            the type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float
            the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        activation_fns: list or object
            the Tensorflow activation function to apply to each layer.  The length of this list should equal
            len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer.
        n_classes: int
            the number of classes
        bypass_layer_sizes: list
            the size of each dense layer in the bypass network. The length of this list determines the number of bypass layers.
        bypass_weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization of bypass layers.
            same requirements as weight_init_stddevs
        bypass_bias_init_consts: list or float
            the value to initialize the biases in bypass layers
            same requirements as bias_init_consts
        bypass_dropouts: list or float
            the dropout probablity to use for bypass layers.
            same requirements as dropouts
        """
        super(self).__init__()
        n_layers = len(layer_sizes)
        n_bypass_layers = len(bypass_layer_sizes)
        self.n_bypass_layers = n_bypass_layers
        self.n_layers = n_layers
        self.n_tasks = n_tasks
        self.n_features = n_features
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers

        if not isinstance(bypass_weight_init_stddevs, SequenceCollection):
            bypass_weight_init_stddevs = [bypass_weight_init_stddevs
                                         ] * n_bypass_layers
        if not isinstance(bypass_bias_init_consts, SequenceCollection):
            bypass_bias_init_consts = [bypass_bias_init_consts
                                      ] * n_bypass_layers
        if not isinstance(bypass_dropouts, SequenceCollection):
            bypass_dropouts = [bypass_dropouts] * n_bypass_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            bypass_activation_fns = [activation_fns] * n_bypass_layers   
        self.n_tasks = n_tasks
        self.n_features = n_features
        self.n_classes = n_classes
        self.activation_fns = [get_activation(i) for i in activation_fns]
        torch.manual_seed(42)
        self.shared_layers = nn.ModuleList()
        in_size = n_features
        # Adding the shared represenation.
        for size, weight_stddev, bias_const, dropout, activation_fn in zip(layer_sizes, weight_init_stddevs, bias_init_consts, dropouts, activation_fns):
            self.layer = nn.Linear(in_size, size)
            nn.init.trunc_normal_(self.layer.weight, 0, weight_stddev)
            if self.layer.bias is not None:
                self.layer.bias = nn.Parameter(torch.full(self.layer.bias.shape, bias_const))
            self.layer.weight_stddev = weight_stddev
            self.layer.bias_const = bias_const
            self.layer.dropout = nn.Dropout(dropout)
            self.layer.layer_act = activation_fn
            self.shared_layers.append(self.layer)
            in_size = size
        # Adding Task specific layers.
        self.bypass_layers = nn.ModuleList()
        for task in range(self.n_tasks):
            self.task_layers = nn.ModuleList()
            in_size = n_features
            for bypass_size, bypass_weight_stddev, bypass_bias_const, bypass_dropout, bypass_activation_fn in zip(bypass_layer_sizes, bypass_weight_init_stddevs, bypass_bias_init_consts, bypass_dropouts, bypass_activation_fns):
                self.layer_task = nn.Linear(in_size, size)
                nn.init.trunc_normal_(self.layer_task.weight, 0, bypass_weight_stddev)
                if self.layer_task.bias is not None:
                    self.layer_task.bias = nn.Parameter(torch.full(self.layer_task.bias.shape, bypass_bias_const))
                self.layer_task.weight_stddev = bypass_weight_stddev
                self.layer_task.bias_const = bypass_bias_const
                self.layer_task.dropout = nn.Dropout(bypass_dropout)
                self.layer_task.layer_act = bypass_activation_fn
                self.task_layers.append(self.layer_task)
                in_size = size
            self.bypass_layers.append(self.task_layers)

        
            
