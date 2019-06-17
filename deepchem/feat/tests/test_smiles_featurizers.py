from unittest import TestCase
import numpy as np
from nose.tools import assert_equals
from deepchem.feat import SmilesToSeq, SmilesToImage
from deepchem.feat.smiles_featurizers import create_char_to_idx
import os


class TestSmilesFeaturizers(TestCase):
  """Tests for SmilesToSeq and SmilesToImage featurizers."""

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
    from rdkit import Chem
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    expected_seq_len = self.feat.max_len + 2 * self.feat.pad_len

    features = self.feat.featurize(mols)
    assert_equals(features.shape[0], len(smiles))
    assert_equals(features.shape[-1], expected_seq_len)

  def test_reconstruct_from_seq(self):
    """Test SMILES reconstruction from features."""
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    features = self.feat.featurize(mols)

    reconstructed_smile = self.feat.smiles_from_seq(features[0])
    assert_equals(smiles[0], reconstructed_smile)
