import numpy as np
import unittest

from deepchem.feat import MordredDescriptors


class TestMordredDescriptors(unittest.TestCase):
  """
  Test MordredDescriptors.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)

  def test_mordred_descriptors(self):
    """
    Test simple descriptors.
    """
    featurizer = MordredDescriptors()
    descriptors = featurizer([self.mol])
    assert descriptors.shape == (1, 1613)
    assert np.allclose(descriptors[0][0:3],
                       np.array([9.54906713, 9.03919229, 1.0]))

  def test_mordred_descriptors_with_3D_info(self):
    """
    Test simple descriptors with 3D info
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    featurizer = MordredDescriptors(ignore_3D=False)
    descriptors = featurizer([self.mol])
    assert descriptors.shape == (1, 1826)
    assert np.allclose(descriptors[0][780:784], np.array([0.0, 0.0, 0.0, 0.0]))

    # calculate coordinates
    mol = self.mol
    mol_with_conf = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_conf, AllChem.ETKDG())
    descriptors = featurizer([mol_with_conf])
    assert descriptors.shape == (1, 1826)
    # not zero values
    assert not np.allclose(descriptors[0][780:784],
                           np.array([0.0, 0.0, 0.0, 0.0]))
