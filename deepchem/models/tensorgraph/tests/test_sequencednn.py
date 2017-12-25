import deepchem as dc
import numpy as np
import unittest

class TestSequenceDNN(unittest.TestCase):

  def test_seq_dnn_init(self):
    """Test SequenceDNN can be initialized."""
    model = dc.models.SequenceDNN(10)
    
  def test_seq_dnn_train(self):
    """Test SequenceDNN training works."""
    X = np.random.rand(10, 1, 4, 50)
    y = np.random.randint(0, 2, size=(10, 1))
    #  # TODO(rbharath): Transform these into useful weights.
    #  #class_weight={
    #  #    True: num_sequences / num_positives,
    #  #    False: num_sequences / num_negatives
    #  #} if not multitask else None,
    dataset = dc.data.NumpyDataset(X, y)
    model = dc.models.SequenceDNN(50)
    model.fit(dataset, "binary_crossentropy", nb_epoch=1)
