"""
Test topological fingerprints.
"""
import unittest

from rdkit import Chem

from deepchem.feat import fingerprints as fp


class TestCircularFingerprint(unittest.TestCase):
  """
    Tests for CircularFingerprint.
    """

  def setUp(self):
    """
        Set up tests.
        """
    self.smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(self.smiles)
    self.engine = fp.CircularFingerprint()

  def test_circular_fingerprints(self):
    """
        Test CircularFingerprint.
        """

    # Passing mols as input argument
    rval_from_mols = self.engine([self.mol])
    assert rval_from_mols.shape == (1, self.engine.size)

    # Passing smiles as input argument
    rval_from_smiles = self.engine(smiles=[self.smiles])
    assert rval_from_smiles.shape == (1, self.engine.size)

  def test_sparse_circular_fingerprints(self):
    """
        Test CircularFingerprint with sparse encoding.
        """
    self.engine = fp.CircularFingerprint(sparse=True)

    # Passing mols as input argument
    rval_from_mols = self.engine([self.mol])
    assert rval_from_mols.shape == (1,)
    assert isinstance(rval_from_mols[0], dict)
    assert len(rval_from_mols[0])

    # Passing smiles as input argument
    rval_from_smiles = self.engine(smiles=[self.smiles])
    assert rval_from_smiles.shape == (1,)
    assert isinstance(rval_from_smiles[0], dict)
    assert len(rval_from_smiles[0])

  def test_sparse_circular_fingerprints_with_smiles(self):
    """
        Test CircularFingerprint with sparse encoding and SMILES for each
        fragment.
        """
    self.engine = fp.CircularFingerprint(sparse=True, smiles=True)

    # Passing mols as input argument
    rval_from_mols = self.engine([self.mol])
    assert rval_from_mols.shape == (1,)
    assert isinstance(rval_from_mols[0], dict)
    assert len(rval_from_mols[0])

    # check for separate count and SMILES entries for each fragment
    for fragment_id, value in rval_from_mols[0].items():
      assert 'count' in value
      assert 'smiles' in value

    # Passing smiles as input argument
    rval_from_smiles = self.engine(smiles=[self.smiles])
    assert rval_from_smiles.shape == (1,)
    assert isinstance(rval_from_smiles[0], dict)
    assert len(rval_from_mols[0])

    # check for separate count and SMILES entries for each fragment
    for fragment_id, value in rval_from_smiles[0].items():
      assert 'count' in value
      assert 'smiles' in value
