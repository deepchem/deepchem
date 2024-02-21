import pytest
import numpy as np
import deepchem as dc
import os
import pickle
try:
    import torch
    from deepchem.models.torch_models.layers import default_dict, TextCNN
    import torch.nn as nn
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass
import shutil

@pytest.mark.torch
def test_textcnn_base():
    model = TextCNN(1, default_dict, 1)
    assert model.seq_length == max(model.kernel_sizes)
    large_length = 500
    model = TextCNN(1, default_dict, large_length)
    assert model.seq_length == large_length
