"""
Test that genomic metrics work.
"""
import unittest
import os

import numpy as np
import deepchem as dc
import tensorflow as tf

LETTERS = "ACGT"

from deepchem.metrics.genomic_metrics import get_motif_scores
from deepchem.metrics.genomic_metrics import get_pssm_scores
from deepchem.metrics.genomic_metrics import in_silico_mutagenesis


class TestGenomicMetrics(unittest.TestCase):
  """
  Tests that genomic metrics work as expected.
  """

  def test_get_motif_scores(self):
    """Check that motif_scores have correct shape."""
    # Encode motif
    motif_name = "TAL1_known4"
    sequences = np.array(["ACGTA", "GATAG", "CGCGC"])
    sequences = dc.utils.genomics.seq_one_hot_encode(sequences, letters=LETTERS)
    # sequences now has shape (3, 4, 5, 1)
    self.assertEqual(sequences.shape, (3, 4, 5, 1))

    motif_scores = get_motif_scores(sequences, [motif_name])
    self.assertEqual(motif_scores.shape, (3, 1, 5))

  def test_get_pssm_scores(self):
    """Test get_pssm_scores returns correct shape."""
    motif_name = "TAL1_known4"
    sequences = np.array(["ACGTA", "GATAG", "CGCGC"])
    sequences = dc.utils.genomics.seq_one_hot_encode(sequences, letters=LETTERS)
    # sequences now has shape (3, 4, 5, 1)
    self.assertEqual(sequences.shape, (3, 4, 5, 1))
    pssm = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    pssm_scores = get_pssm_scores(sequences, pssm)
    self.assertEqual(pssm_scores.shape, (3, 5))

  def create_model_for_mutagenesis(self):
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(1, 15, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, 15, activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    return dc.models.KerasModel(keras_model,
                                dc.models.losses.BinaryCrossEntropy())

  def test_in_silico_mutagenesis_shape(self):
    """Test in-silico mutagenesis returns correct shape."""
    # Construct and train SequenceDNN model
    sequences = np.array(["ACGTA", "GATAG", "CGCGC"])
    sequences = dc.utils.genomics.seq_one_hot_encode(sequences, letters=LETTERS)
    labels = np.array([1, 0, 0])
    labels = np.reshape(labels, (3, 1))
    self.assertEqual(sequences.shape, (3, 4, 5, 1))

    #X = np.random.rand(10, 1, 4, 50)
    #y = np.random.randint(0, 2, size=(10, 1))
    #dataset = dc.data.NumpyDataset(X, y)
    dataset = dc.data.NumpyDataset(sequences, labels)
    model = self.create_model_for_mutagenesis()
    model.fit(dataset, nb_epoch=1)

    # Call in-silico mutagenesis
    mutagenesis_scores = in_silico_mutagenesis(model, sequences)
    self.assertEqual(mutagenesis_scores.shape, (1, 3, 4, 5, 1))

  def test_in_silico_mutagenesis_nonzero(self):
    """Test in-silico mutagenesis returns nonzero output."""
    # Construct and train SequenceDNN model
    sequences = np.array(["ACGTA", "GATAG", "CGCGC"])
    sequences = dc.utils.genomics.seq_one_hot_encode(sequences, letters=LETTERS)
    labels = np.array([1, 0, 0])
    labels = np.reshape(labels, (3, 1))
    self.assertEqual(sequences.shape, (3, 4, 5, 1))

    #X = np.random.rand(10, 1, 4, 50)
    #y = np.random.randint(0, 2, size=(10, 1))
    #dataset = dc.data.NumpyDataset(X, y)
    dataset = dc.data.NumpyDataset(sequences, labels)
    model = self.create_model_for_mutagenesis()
    model.fit(dataset, nb_epoch=1)

    # Call in-silico mutagenesis
    mutagenesis_scores = in_silico_mutagenesis(model, sequences)
    self.assertEqual(mutagenesis_scores.shape, (1, 3, 4, 5, 1))

    # Check nonzero elements exist
    assert np.count_nonzero(mutagenesis_scores) > 0
