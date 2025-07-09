import unittest
from deepchem.feat import ACTINNFeaturizer
import pandas as pd
import numpy as np
import os


class TestACTINNFeaturizer(unittest.TestCase):

    def test_actinn_compare_with_existing_implementation(self):
        dataset = os.path.join(os.path.dirname(__file__), "data",
                               "sc_rna_seq_data")
        ref_array = np.load(
            os.path.join(dataset, 'actinn_feat_ref_compressed.npz'))['arr_0']
        train_set = pd.read_hdf(os.path.join(dataset, 'train_set.h5'))
        featurizer = ACTINNFeaturizer()
        train_dataset = featurizer.featurize(train_set)
        assert np.allclose(train_dataset.X, ref_array, rtol=1e-5, atol=1e-6)
