import deepchem as dc
import numpy as np
import unittest

class TestSequenceDNN(unittest.TestCase):

  def test_sequence_dnn_init(self):
    """Test SequenceDNN can be initialized."""
    model = dc.models.SequenceDNN(10)
    
