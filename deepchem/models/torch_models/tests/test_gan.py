import deepchem as dc
import numpy as np
import pytest
import tempfile
from flaky import flaky

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ExampleGAN(dc.models.GAN):
        
        def get_noise_input_shape(self):
            return (2,)
        
        def get_data_input_shapes(self):
            return [(1,)]
        
        def get_conditional_input_shapes(self):
            return [(1,)]
        
        
        
    has_torch = True
except:
    has_torch = False