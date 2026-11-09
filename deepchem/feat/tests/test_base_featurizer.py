import unittest

import deepchem as dc


class _FailingFeaturizer(dc.feat.Featurizer):
    """Featurizer that always fails."""

    def _featurize(self, datapoint, **kwargs):
        raise ValueError("cannot featurize")


class TestFeaturizerBase(unittest.TestCase):
    """
    Tests for the Featurizer base class.
    """

    def test_failed_datapoint_warning_has_index(self):
        """
        The warning for a failed datapoint should report its index.
        """
        featurizer = _FailingFeaturizer()
        with self.assertLogs("deepchem.feat.base_classes",
                             level="WARNING") as logs:
            out = featurizer.featurize(["a", "b"])
        message = "\n".join(logs.output)
        assert "Failed to featurize datapoint 0." in message
        assert "Failed to featurize datapoint 1." in message
        assert "%d" not in message
        assert out.shape[0] == 2
