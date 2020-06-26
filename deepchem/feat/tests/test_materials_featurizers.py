"""
Test featurizers for inorganic crystals.
"""
import numpy as np
import unittest

from deepchem.feat.materials_featurizers import ElementPropertyFingerprint, SineCoulombMatrix, StructureGraphFeaturizer


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

  def test_element_property_fingerprint(self):
    """
    Test Element Property featurizer.
    """

    featurizer = ElementPropertyFingerprint(data_source='matminer')
    features = featurizer.featurize([self.formula])

    assert len(features[0]) == 65
    assert np.allclose(
        features[0][:5], [2.16, 2.58, 0.42, 2.44, 0.29698485], atol=0.1)

  def test_sine_coulomb_matrix(self):
    """
    Test SCM featurizer.
    """

    featurizer = SineCoulombMatrix(max_atoms=1)
    features = featurizer.featurize([self.struct_dict])

    assert len(features) == 1
    assert np.isclose(features[0], 1244, atol=.5)

  def test_structure_graph_featurizer(self):
    """
    Test StructureGraphFeaturizer.
    """

    featurizer = StructureGraphFeaturizer(radius=3.0, max_neighbors=6)
    features = featurizer.featurize([self.struct_dict])

    assert len(features[0]) == 3
    assert features[0][0] == 26
    assert features[0][1].shape == (6, 16)
