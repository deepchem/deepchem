import unittest
from deepchem.feat import RobertaFeaturizer
import pytest


class TestRobertaFeaturizer(unittest.TestCase):
  """Tests for RobertaFeaturizer"""

  @pytest.mark.torch
  def setUp(self):
    self.smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    self.long_molecule_smiles = [
        "CCCCCCCCCCCCCCCCCCCC(=O)OCCCNC(=O)c1ccccc1SSc1ccccc1C(=O)NCCCOC(=O)CCCCCCCCCCCCCCCCCCC"
    ]
    self.featurizer = RobertaFeaturizer.from_pretrained(
        "seyonec/SMILES_tokenized_PubChem_shard00_160k")

  @pytest.mark.torch
  def test_smiles_call(self):
    """Test __call__ method for the featurizer, which is inherited from HuggingFace's RobertaTokenizerFast"""
    embedding = self.featurizer(
        self.smiles, add_special_tokens=True, truncation=True)
    embedding_long = self.featurizer(
        self.long_molecule_smiles * 2, add_special_tokens=True, truncation=True)
    for emb in [embedding, embedding_long]:
      assert 'input_ids' in emb.keys() and 'attention_mask' in emb.keys()
      assert len(emb['input_ids']) == 2 and len(emb['attention_mask']) == 2

  @pytest.mark.torch
  def test_smiles_featurize(self):
    """Test the .featurize method, which will convert the dictionary output to an array

    Checks that all SMILES are featurized and that each featurization
    contains input_ids and attention_mask
    """
    feats = self.featurizer.featurize(
        self.smiles, add_special_tokens=True, truncation=True)
    assert (len(feats) == 2)
    assert (all([len(f) == 2 for f in feats]))

    long_feat = self.featurizer.featurize(
        self.long_molecule_smiles, add_special_tokens=True, truncation=True)
    assert (len(long_feat) == 1)
    assert (len(long_feat[0] == 2))
