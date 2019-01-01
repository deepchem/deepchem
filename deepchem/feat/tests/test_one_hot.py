from unittest import TestCase

import deepchem as dc
from nose.tools import assert_equals


class TestOneHotFeaturizer(TestCase):
  """Tests for the one-hot featurizer."""

  def test_featurize(self):
    from rdkit import Chem
    smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    featurizer = dc.feat.one_hot.OneHotFeaturizer(dc.feat.one_hot.zinc_charset)
    one_hots = featurizer.featurize(mols)
    untransformed = featurizer.untransform(one_hots)
    assert_equals(len(smiles), len(untransformed))
    for i in range(len(smiles)):
      assert_equals(smiles[i], untransformed[i][0])
