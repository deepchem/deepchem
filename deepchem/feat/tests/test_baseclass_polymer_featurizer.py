from deepchem.feat.base_classes import PolymerFeaturizer
import unittest
import numpy as np


class DummyPolymerFeatClass(PolymerFeaturizer):
    """
    Dummy polymer class derived from PolymerFeaturizer
    """

    def __init__(self):
        pass

    def _featurize(self, datapoint, **kwargs):
        """
        Returns string object from recieved datapoint as string, int, float, etc
        """
        return datapoint


class SampleObject():

    def __init__(self, smiles):
        self.smiles = smiles


class DummyPolymerFeatObjClass(PolymerFeaturizer):
    """
    Dummy polymer class derived from PolymerFeaturizer that returns object
    """

    def __init__(self):
        pass

    def _featurize(self, datapoint, **kwargs):
        """
        Returns string object from recieved datapoint as string, int, float, etc
        """
        datapoint = SampleObject(datapoint)
        return datapoint


class TestPolymerFeatClass(unittest.TestCase):
    """
    Test of dummy polymer featurizer
    """

    def setUp(self):
        """
        Set up test case.
        """
        self.polymer_feat = DummyPolymerFeatClass()

    def test_valid_featurize(self):
        """
        Test featurization of dummy polymer class
        """
        datapoint = 'CCC'
        datapoints = ['CCC']
        features = self.polymer_feat.featurize(datapoint)
        assert isinstance(features, np.ndarray)
        _ = self.polymer_feat.featurize(datapoints)

    def test_invalid_featurize(self):
        """
        Test featurization of invalid input type
        """
        datapoint = 69
        with self.assertRaises(TypeError):
            _ = self.polymer_feat.featurize(datapoint)

        datapoints = ["CCC", 69]
        featured = self.polymer_feat.featurize(datapoints)
        assert featured[1].shape == (0,)


class TestPolymerFeatObjClass(unittest.TestCase):
    """
    Test of dummy polymer featurizer that returns objects
    """

    def setUp(self):
        """
        Set up test case.
        """
        self.polymer_feat = DummyPolymerFeatObjClass()

    def test_obj_return(self):
        """
        Test featurization of dummy polymer class that returns object
        """
        datapoint = 'CCC'
        features = self.polymer_feat.featurize(datapoint)
        assert features.dtype == 'object'
        assert isinstance(features[0], SampleObject)
