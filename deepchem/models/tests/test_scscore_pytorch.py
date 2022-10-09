import pytest
import numpy as np

import unittest

try:
  import torch
  import torch.nn as nn
  from deepchem.models.torch_models.scscore import ScScore
  has_torch = True
except:
  has_torch = False
  
  
@unittest.skipIf(not has_torch, 'torch is not installed')
@pytest.mark.torch
def test_scscore_pytorch():
    """
    This test evaluates the ScScore.
    """
    
    #Initializes the ScScore
    n_features = 1024
    layer_sizes = [300,300,300,300,300]
    dropout_prob = 0
    model = ScScore(n_features= n_features, layer_sizes= layer_sizes, dropout = dropout_prob)
    
    batch_size = 32
    test_input = torch.randn((batch_size,n_features))
    
    assert model(test_input).shape == torch.Size([batch_size,1]) , "Model output doesn't match the expected shape"
