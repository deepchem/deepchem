import unittest
from unittest.mock import patch
import deepchem as dc
from deepchem.molnet.load_function.uspto_datasets import load_uspto


class TestUSPTOLoader(unittest.TestCase):
    """Test featurizer resolution in load_uspto loader."""

    @patch('deepchem.molnet.load_function.uspto_datasets._USPTOLoader')
    def test_load_uspto_custom_featurizer(self, mock_loader_cls):
        """Test passing custom Featurizer instance to load_uspto."""
        instance = mock_loader_cls.return_value
        instance.name = 'USPTO_MIT'
        instance.load_dataset.return_value = ([], (), [])

        dummy = dc.feat.DummyFeaturizer()
        load_uspto(featurizer=dummy)
        passed_feat = mock_loader_cls.call_args[0][0]
        self.assertIsInstance(passed_feat, dc.feat.DummyFeaturizer)

    @patch('deepchem.molnet.load_function.uspto_datasets._USPTOLoader')
    def test_load_uspto_plain_featurizer(self, mock_loader_cls):
        """Test 'plain' string featurizer resolves to DummyFeaturizer."""
        instance = mock_loader_cls.return_value
        instance.name = 'USPTO_MIT'
        instance.load_dataset.return_value = ([], (), [])

        load_uspto(featurizer="plain")
        passed_feat = mock_loader_cls.call_args[0][0]
        self.assertIsInstance(passed_feat, dc.feat.DummyFeaturizer)

    @patch('deepchem.molnet.load_function.uspto_datasets._USPTOLoader')
    def test_load_uspto_shortcut_featurizer(self, mock_loader_cls):
        """Test standard shortcut string resolves properly."""
        instance = mock_loader_cls.return_value
        instance.name = 'USPTO_MIT'
        instance.load_dataset.return_value = ([], (), [])

        load_uspto(featurizer="raw")
        passed_feat = mock_loader_cls.call_args[0][0]
        self.assertIsInstance(passed_feat, dc.feat.RawFeaturizer)

    def test_load_uspto_invalid_featurizer(self):
        """Test invalid featurizer string raises ValueError."""
        with self.assertRaises(ValueError):
            load_uspto(featurizer="invalid_nonexistent")
