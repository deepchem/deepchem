"""
Tests for BBBC Loaders.
"""

import unittest
import deepchem as dc


class TestBBBCLoader(unittest.TestCase):
    """
    Test BBBC Loaders
    """

    def test_bbbc001(self):
        """
        Test loading BBBC001
        """
        loader = dc.molnet.load_bbbc001()
        tasks, dataset, transformers = loader
        train, val, test = dataset
        assert train.X.shape == (4, 512, 512)
        assert train.y.shape == (4,)
        assert train.w.shape == (4,)
        assert train.ids.shape == (4,)
        assert val.X.shape == (1, 512, 512)
        assert val.y.shape == (1,)
        assert val.w.shape == (1,)
        assert val.ids.shape == (1,)
        assert test.X.shape == (1, 512, 512)
        assert test.y.shape == (1,)
        assert test.w.shape == (1,)
        assert test.ids.shape == (1,)

    def test_bbbc002(self):
        """
        Test loading BBBC002
        """
        loader = dc.molnet.load_bbbc002()
        tasks, dataset, transformers = loader
        train, val, test = dataset
        assert train.X.shape == (40, 512, 512)
        assert train.y.shape == (40,)
        assert train.w.shape == (40,)
        assert train.ids.shape == (40,)
        assert val.X.shape == (5, 512, 512)
        assert val.y.shape == (5,)
        assert val.w.shape == (5,)
        assert val.ids.shape == (5,)
        assert test.X.shape == (5, 512, 512)
        assert test.y.shape == (5,)
        assert test.w.shape == (5,)
        assert test.ids.shape == (5,)

    def test_bbbc004_segmentation(self):
        """
        Test loading BBBC004 Segmentation Masks as labels
        """
        loader = dc.molnet.load_bbbc004(load_segmentation_masks=True)
        tasks, dataset, transformers = loader
        train, val, test = dataset
        assert train.X.shape == (16, 950, 950)
        assert train.y.shape == (16, 950, 950, 3)
        assert train.w.shape == (16, 1)
        assert train.ids.shape == (16,)
        assert val.X.shape == (2, 950, 950)
        assert val.y.shape == (2, 950, 950, 3)
        assert val.w.shape == (2, 1)
        assert val.ids.shape == (2,)
        assert test.X.shape == (2, 950, 950)
        assert test.y.shape == (2, 950, 950, 3)
        assert test.w.shape == (2, 1)
        assert test.ids.shape == (2,)

    def test_bbbc004_counts(self):
        """
        Test loading BBBC004 Cell Counts as labels
        """
        loader = dc.molnet.load_bbbc004()
        tasks, dataset, transformers = loader
        train, val, test = dataset
        assert train.X.shape == (16, 950, 950)
        assert train.y.shape == (16,)
        assert train.w.shape == (16,)
        assert train.ids.shape == (16,)
        assert val.X.shape == (2, 950, 950)
        assert val.y.shape == (2,)
        assert val.w.shape == (2,)
        assert val.ids.shape == (2,)
        assert test.X.shape == (2, 950, 950)
        assert test.y.shape == (2,)
        assert test.w.shape == (2,)
        assert test.ids.shape == (2,)
