import unittest
import numpy as np
import deepchem as dc
from deepchem.splits.ave_splitter import AVESplitter


class TestAVESplitter(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic dataset
        np.random.seed(123)
        X = np.random.rand(100, 10)
        # Binary labels
        y = np.random.randint(0, 2, size=(100, 1))

        self.dataset = dc.data.NumpyDataset(X, y)

    def test_ave_split(self):
        # Small max_iter to make the test run quickly
        splitter = AVESplitter(metric='euclidean', max_iter=5, pop_size=10, next_gen_size=5)

        train_inds, valid_inds, test_inds = splitter.split(
            self.dataset,
            frac_train=0.8,
            frac_valid=0.2,
            frac_test=0.0
        )

        # Verify sizes roughly
        # Note: AVE Genetic Algorithm can fluctuate slightly in size, but it should be close
        self.assertGreater(len(train_inds), 0)
        self.assertGreater(len(valid_inds), 0)
        self.assertEqual(len(test_inds), 0)

        # Verify no overlap
        intersect = set(train_inds).intersection(set(valid_inds))
        self.assertEqual(len(intersect), 0)

        # Verify all elements in valid bounds
        self.assertTrue(all([0 <= i < len(self.dataset) for i in train_inds]))
        self.assertTrue(all([0 <= i < len(self.dataset) for i in valid_inds]))

    def test_ave_split_with_test(self):
        splitter = AVESplitter(metric='euclidean', max_iter=5, pop_size=10, next_gen_size=5)

        train_inds, valid_inds, test_inds = splitter.split(
            self.dataset,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1
        )

        self.assertGreater(len(train_inds), 0)
        self.assertGreater(len(valid_inds), 0)
        self.assertGreater(len(test_inds), 0)

        intersect_tv = set(train_inds).intersection(set(valid_inds))
        intersect_tt = set(train_inds).intersection(set(test_inds))
        intersect_vt = set(valid_inds).intersection(set(test_inds))

        self.assertEqual(len(intersect_tv), 0)
        self.assertEqual(len(intersect_tt), 0)
        self.assertEqual(len(intersect_vt), 0)


if __name__ == '__main__':
    unittest.main()
