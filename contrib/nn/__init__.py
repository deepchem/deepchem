"""
Imports a number of useful deep learning primitives into one place.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deepchem.nn.model_ops import weight_decay
from deepchem.nn.model_ops import optimizer
from deepchem.nn.model_ops import add_bias
from deepchem.nn.model_ops import fully_connected_layer
from deepchem.nn.model_ops import multitask_logits
from deepchem.nn.model_ops import softmax_N
from deepchem.nn.objectives import mean_squared_error
