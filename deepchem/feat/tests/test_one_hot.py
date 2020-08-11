import unittest
from deepchem.feat import OneHotFeaturizer


class TestOneHotFeaturizer(unittest.TestCase):
  """Tests for the one-hot featurizer."""

  def test_featurize(self):
    from rdkit import Chem
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    featurizer = OneHotFeaturizer()
    one_hots = featurizer.featurize(mols)
    untransformed = featurizer.untransform(one_hots)
    assert len(smiles) == len(untransformed)
    for i in range(len(smiles)):
      assert smiles[i] == untransformed[i]
