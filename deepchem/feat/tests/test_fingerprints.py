"""
Test topological fingerprints.
"""
import unittest
from deepchem.feat import fingerprints as fp


class TestCircularFingerprint(unittest.TestCase):
  """
  Tests for CircularFingerprint.
  """

  def setUp(self):
    """
    Set up tests.
    """
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    from rdkit import Chem
    self.mol = Chem.MolFromSmiles(smiles)
    self.engine = fp.CircularFingerprint()

  def test_circular_fingerprints(self):
    """
    Test CircularFingerprint.
    """
    rval = self.engine([self.mol])
    assert rval.shape == (1, self.engine.size)

  def test_circular_fingerprints_on_smiles(self):
    """
    Test CircularFingerprint on smiles
    """
    rval = self.engine('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert rval.shape == (1, self.engine.size)

    rval = self.engine(['CC(=O)OC1=CC=CC=C1C(=O)O'])
    assert rval.shape == (1, self.engine.size)

  def test_sparse_circular_fingerprints(self):
    """
    Test CircularFingerprint with sparse encoding.
    """
    self.engine = fp.CircularFingerprint(sparse=True)
    rval = self.engine([self.mol])
    assert rval.shape == (1,)
    assert isinstance(rval[0], dict)
    assert len(rval[0])

  def test_sparse_circular_fingerprints_with_smiles(self):
    """
    Test CircularFingerprint with sparse encoding and SMILES for each
    fragment.
    """
    self.engine = fp.CircularFingerprint(sparse=True, smiles=True)
    rval = self.engine([self.mol])
    assert rval.shape == (1,)
    assert isinstance(rval[0], dict)
    assert len(rval[0])

    # check for separate count and SMILES entries for each fragment
    for fragment_id, value in rval[0].items():
      assert 'count' in value
      assert 'smiles' in value
