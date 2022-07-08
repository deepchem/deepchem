import unittest
import numpy as np
from rdkit import Chem
from deepchem.feat.molecule_featurizers.dmpnn_featurizer import generate_global_features


class TestGlobalFeatureGenerator(unittest.TestCase):
  """
  Test for `generate_global_features` helper function which generates global features for DMPNN featurizer
  """

  def setUp(self):
    """
    Set up tests.
    """
    smiles_list = ["C", "[H]"]
    self.mol = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    self.feature_generators = [[''], ['morgan'], ['morgan', ''],
                               ['morgan', 'morgan']]

  def test_generator_invalid_name(self):
    """
    Test for generator when given name of feature generator is not in the list of available generators
    """
    global_features = generate_global_features(self.mol[0],
                                               self.feature_generators[0])
    assert (global_features == np.empty(0)).all()

  def test_generator_morgan(self):
    """
    Test for generator when 'morgan' feature generator is provided
    """
    global_features = generate_global_features(self.mol[0],
                                               self.feature_generators[1])
    assert len(global_features) == 2048

    nonzero_features_indicies = global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 1
    assert nonzero_features_indicies[0] == 1264
    assert global_features[nonzero_features_indicies[0]] == 1.0

  def test_generator_morgan_with_invalid_name(self):
    """
    Test for generator when 'morgan' feature generator and an unavailable generator name is provided
    """
    global_features = generate_global_features(self.mol[0],
                                               self.feature_generators[2])
    assert len(global_features) == 2048

    nonzero_features_indicies = global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 1
    assert nonzero_features_indicies[0] == 1264
    assert global_features[nonzero_features_indicies[0]] == 1.0

  def test_generator_morgan_twice(self):
    """
    Test for generator when names of multiple generators are provided
    """
    global_features = generate_global_features(self.mol[0],
                                               self.feature_generators[3])
    assert len(global_features) == 4096

    nonzero_features_indicies = global_features.nonzero()[0]
    assert len(nonzero_features_indicies) == 2
    assert nonzero_features_indicies[0] == 1264
    assert nonzero_features_indicies[1] == 1264 + 2048
    assert global_features[nonzero_features_indicies[0]] == 1.0
    assert global_features[nonzero_features_indicies[1]] == 1.0

  def test_generator_hydrogen(self):
    """
    Test for generator when provided RDKit mol contains only Hydrogen atoms
    """
    global_features = generate_global_features(self.mol[1],
                                               self.feature_generators[2])
    assert (global_features == np.zeros(2048)).all()
