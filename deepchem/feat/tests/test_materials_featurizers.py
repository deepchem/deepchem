"""
Test featurizers for inorganic crystals.
"""
import numpy as np
import unittest

from deepchem.feat.materials_featurizers import ChemicalFingerprint, SineCoulombMatrix, StructureGraphFeaturizer


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
    features = featurizer.featurize([self.formula])

    assert len(features[0]) == 65
    assert np.allclose(
        features[0][:5], [2.16, 2.58, 0.42, 2.44, 0.29698485], atol=0.1)

  def testSCM(self):
    """
    Test SCM featurizer.
    """

    featurizer = SineCoulombMatrix(1)
    features = featurizer.featurize([self.struct_dict])

    assert len(features) == 1
    assert np.isclose(features[0], 1244, atol=.5)

  def testSGF(self):
    """
    Test StructureGraphFeaturizer.
    """

    featurizer = StructureGraphFeaturizer()
    features = featurizer.featurize([self.struct_dict])

    assert len(features[0]) == 3
    assert features[0][0] == 26
    assert len(features[0][2]) == 6
