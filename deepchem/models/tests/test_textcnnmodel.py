import unittest
import pytest
try:
    import tensorflow as tf  # noqa: F401
    from deepchem.models import TextCNNModel
    from deepchem.models.text_cnn import default_dict
    has_tensorflow = True
except:
    has_tensorflow = False


class TestTextCNNModel(unittest.TestCase):

    @pytest.mark.tensorflow
    def test_set_length(self):
        model = TextCNNModel(1, default_dict, 1)
        self.assertEqual(model.seq_length, max(model.kernel_sizes))

        large_length = 500
        model = TextCNNModel(1, default_dict, large_length)
        self.assertEqual(model.seq_length, large_length)
