import torch
import torch.nn as nn

import logging
from deepchem.models.torch_models import TorchModel

logger = logging.getLogger(__name__)

from collections.abc import Sequence as SequenceCollection

class ProgressiveMultitask(nn.Module):

    def __init__(self,
                 n_tasks,
                 n_features,
                 alpha_init_stddevs=0.02,
                 layer_sizes=[1000],
                 weight_init_stddevs=0.02,
                 bias_init_consts=1.0,
                 weight_decay_penalty=0.0,
                 weight_decay_penalty_type="l2",
                 activation_fns = nn.ReLU,
                 dropouts=0.5,
                 n_outputs=1,
                **kwargs):
        
        if weight_decay_penalty != 0.0:
            raise ValueError('Weight decay is not currently supported')
        self.n_tasks = n_tasks
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.alpha_init_stddevs = alpha_init_stddevs
        self.weight_init_stddevs = weight_init_stddevs
        self.bias_init_consts = bias_init_consts
        self.dropouts = dropouts
        self.activation_fns = activation_fns
        self.n_outputs = n_outputs

        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, SequenceCollection):
            self.weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(alpha_init_stddevs, SequenceCollection):
            self.alpha_init_stddevs = [alpha_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            self.bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            self.dropouts = [dropouts] * n_layers
        if not isinstance(activation_fns, SequenceCollection):
            self.activation_fns = [activation_fns] * n_layers

        super(ProgressiveMultitask, self).__init__()

        self.layers = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.alphas = nn.ParameterList()

        for task in range(n_tasks):
            layer_list = []
            adapter_list = []
            alpha_list = []
            prev_size = n_features
            for i, (size, dropout, activation_fn) in enumerate(zip(self.layer_sizes, self.dropouts, self.activation_fns)):
                layer = []
                layer.append(self._init_linear(prev_size, size, i))
                layer.append(activation_fn())
                if dropout > 0:
                    layer.append(nn.Dropout(dropout))

                layer_list.append(nn.Sequential(*layer))

                if task > 0:
                    if i > 0:
                        adapter, alpha = self._get_adapter(task, prev_size, size, i)
                        adapter_list.append(adapter)
                        alpha_list.append(alpha)
                    
                prev_size = size
            
            layer_list.append(nn.Sequential(self._init_linear(prev_size, n_outputs, len(self.layer_sizes))))
            self.layers.append(nn.ModuleList(layer_list))
            if task > 0:
                adapter, alpha = self._get_adapter(task, prev_size, n_outputs, len(self.layer_sizes))
                adapter_list.append(adapter)
                alpha_list.append(alpha)
                self.adapters.append(nn.ModuleList(adapter_list))
                self.alphas.append(nn.ParameterList(alpha_list))

    def _get_adapter(self, task, prev_size, size, layer_num):
        adapter = nn.Sequential(
                            self._init_linear(prev_size * task, prev_size, layer_num),
                            # activation_fn(), # add it? not present in tf implementation
                            self._init_linear(prev_size, size, layer_num)
                        ) 
        alpha_init_stddev = self.alpha_init_stddevs[layer_num] if layer_num < len(self.layer_sizes) else self.alpha_init_stddevs[-1]
        alpha = torch.empty(1, requires_grad=True)
        nn.init.trunc_normal_(alpha, std=alpha_init_stddev)
        return adapter, alpha

    def _init_linear(self, in_features, out_features, layer_num):
        if layer_num < len(self.layer_sizes):
            weight_init_stddev = self.weight_init_stddevs[layer_num]
            bias_init_const = self.bias_init_consts[layer_num]
        elif layer_num == len(self.layer_sizes):
            weight_init_stddev = self.weight_init_stddevs[-1]
            bias_init_const = self.bias_init_consts[-1]

        layer = nn.Linear(in_features, out_features)
        nn.init.trunc_normal_(layer.weight, std=weight_init_stddev)
        nn.init.constant_(layer.bias, bias_init_const)

        return layer

    def forward(self, x):
        outputs = []
        logits = []
        for task in range(self.n_tasks):
            x_ = x
            layer_outputs = []
            for i, layer in enumerate(self.layers[task]):
                x_ = layer(x_)
                if task > 0 and i > 0:
                    adapter = self.adapters[task-1][i-1]
                    alpha = self.alphas[task-1][i-1]
                    prev_logits = [logits[t][i - 1] for t in range(task)]
                    adapter_input = torch.cat(prev_logits, dim=-1)
                    adapter_input = alpha * adapter_input
                    x_ = x_ + adapter(adapter_input)

                layer_outputs.append(x_)

            logits.append(layer_outputs)
            outputs.append(x_)
        
        return torch.stack(outputs, dim=1)
    
# sample = torch.randn(64, 10)

# model = ProgressiveMultitask(12, 10, 0.02, [1000, 450])
# output = model(sample)

# print(output.shape)