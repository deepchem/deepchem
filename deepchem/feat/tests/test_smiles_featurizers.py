import os
import unittest

import numpy as np

from deepchem.feat import create_char_to_idx, SmilesToSeq, SmilesToImage


class TestSmilesToSeq(unittest.TestCase):
  """Tests for SmilesToSeq featurizers."""

  def setUp(self):
    """Setup."""
    pad_len = 5
    max_len = 35
    filename = os.path.join(
        os.path.dirname(__file__), "data", "chembl_25_small.csv")
    char_to_idx = create_char_to_idx(filename, max_len=max_len)
    self.feat = SmilesToSeq(
        char_to_idx=char_to_idx, max_len=max_len, pad_len=pad_len)

  def test_smiles_to_seq_featurize(self):
    """Test SmilesToSeq featurization."""
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    expected_seq_len = self.feat.max_len + 2 * self.feat.pad_len

    features = self.feat.featurize(smiles)
    assert features.shape[0] == len(smiles)
    assert features.shape[-1] == expected_seq_len

  def test_reconstruct_from_seq(self):
    """Test SMILES reconstruction from features."""
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
    features = self.feat.featurize(smiles)
    # not support array style inputs
    reconstructed_smile = self.feat.smiles_from_seq(features[0])
    assert smiles[0] == reconstructed_smile


class TestSmilesToImage(unittest.TestCase):
  """Tests for SmilesToImage featurizers."""

  def setUp(self):
    """Setup."""
    self.smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]

  def test_smiles_to_image(self):
    """Test default SmilesToImage"""
    featurizer = SmilesToImage()
    features = featurizer.featurize(self.smiles)
    assert features.shape == (2, 80, 80, 1)

  def test_smiles_to_image_with_res(self):
    """Test SmilesToImage with res"""
    featurizer = SmilesToImage()
    base_features = featurizer.featurize(self.smiles)
    featurizer = SmilesToImage(res=0.6)
    features = featurizer.featurize(self.smiles)
    assert features.shape == (2, 80, 80, 1)
    assert not np.allclose(base_features, features)

  def test_smiles_to_image_with_image_size(self):
    """Test SmilesToImage with image_size"""
    featurizer = SmilesToImage(img_size=100)
    features = featurizer.featurize(self.smiles)
    assert features.shape == (2, 100, 100, 1)

  def test_smiles_to_image_with_max_len(self):
    """Test SmilesToImage with max_len"""
    smiles_length = [len(s) for s in self.smiles]
    assert smiles_length == [26, 25]
    featurizer = SmilesToImage(max_len=25)
    features = featurizer.featurize(self.smiles)
    assert features[0].shape == (0,)
    assert features[1].shape == (80, 80, 1)

  def test_smiles_to_image_with_img_spec(self):
    """Test SmilesToImage with img_spec"""
    featurizer = SmilesToImage()
    base_features = featurizer.featurize(self.smiles)
    featurizer = SmilesToImage(img_spec='engd')
    features = featurizer.featurize(self.smiles)
    assert features.shape == (2, 80, 80, 4)
    assert not np.allclose(base_features, features)
