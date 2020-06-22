"""
Test featurizers for inorganic crystals.
"""
import numpy as np
import unittest

from deepchem.feat.materials_featurizers import ChemicalFingerprint, SineCoulombMatrix


class TestMaterialFeaturizers(unittest.TestCase):
  """
  Test material featurizers.
  """

  def setUp(self):
    """
    Set up tests.
    """
    self.formula = 'MoS2'
    self.struct_dict = {
        '@module':
        'pymatgen.core.structure',
        '@class':
        'Structure',
        'charge':
        None,
        'lattice': {
            'matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'a': 1.0,
            'b': 1.0,
            'c': 1.0,
            'alpha': 90.0,
            'beta': 90.0,
            'gamma': 90.0,
            'volume': 1.0
        },
        'sites': [{
            'species': [{
                'element': 'Fe',
                'occu': 1
            }],
            'abc': [0.0, 0.0, 0.0],
            'xyz': [0.0, 0.0, 0.0],
            'label': 'Fe',
            'properties': {}
        }]
    }

  def testCF(self):
    """
    Test CF featurizer.
    """

    featurizer = ChemicalFingerprint(data_source='matminer')

    assert isinstance(featurizer, ChemicalFingerprint)

  def testSCM(self):
    """
    Test SCM featurizer.
    """

    featurizer = SineCoulombMatrix(1)
    features = featurizer.featurize([self.struct_dict])

    assert np.isclose(features[0], 1244, atol=.5)
