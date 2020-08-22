# Requriments - transformers, tokenizers

from unittest import TestCase
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from transformers import RobertaForMaskedLM


class TestSmilesTokenizer(TestCase):
  """Tests the SmilesTokenizer to load the USPTO vocab file and a ChemBERTa Masked LM model with pre-trained weights.."""


  def test_featurize(self):
    from rdkit import Chem
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    featurizer = dc.feat.one_hot.OneHotFeaturizer(dc.feat.one_hot.zinc_charset)
    one_hots = featurizer.featurize(mols)
    untransformed = featurizer.untransform(one_hots)
    assert len(smiles) == len(untransformed)
    for i in range(len(smiles)):
      assert smiles[i] == untransformed[i][0]
