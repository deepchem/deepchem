import unittest
from deepchem.feat.molecule_featurizers import SparseMatrixOneHotFeaturizer


class TestSparseMatrixOneHotFeaturizer(unittest.TestCase):
    """
    Test SparseMatrixOneHotFeaturizer.
    """

    def test_sparsemat_arbitrary_default_charset(self):
        """
        Test simple one hot encoding
        """
        featurizer = SparseMatrixOneHotFeaturizer()
        sequence = "MMMQLA"
        encodings = featurizer.featurize([sequence])
        assert encodings[0].shape[0] == 6
        assert encodings[0].shape[1] == 25

    def test_sparsemat_arbitrary_default_charset_utransform(self):
        """
        Test simple one hot encoding
        """
        featurizer = SparseMatrixOneHotFeaturizer()
        sequence = "MMMQLA"
        encodings = featurizer.featurize([sequence])
        out = featurizer.untransform(encodings[0])
        assert out == "MMMQLA"

    def test_sparsemat_arbitrary_arbitrary_charset(self):
        """
        Test simple one hot encoding
        """
        charset = ["A", "B", "C"]
        featurizer = SparseMatrixOneHotFeaturizer(charset)
        sequence = "AAAB"
        encodings = featurizer.featurize([sequence])
        array = encodings[0].toarray()
        assert encodings[0].shape[0] == 4
        assert encodings[0].shape[1] == 3
        assert array[0][0] == 1
        assert array[0][1] == 0

    def test_sparsemat_arbitrary_unkonw_val(self):
        """
        Test simple one hot encoding
        """
        charset = ["A", "B", "C"]
        featurizer = SparseMatrixOneHotFeaturizer(charset)
        sequence = "AAAD"
        encodings = featurizer.featurize([sequence])
        array = encodings[0].toarray()
        assert encodings[0].shape[0] == 4
        assert encodings[0].shape[1] == len(charset)
        assert array[0][0] == 1
        assert array[-1][-1] == 0
