import unittest


from deepchem.utils.graph_conv_utils import one_hot_encode


# TODO: add more test cases
class TestGraphConvUtils(unittest.TestCase):
  def test_one_hot_encode(self):
    # string set
    assert one_hot_encode("a", ["a", "b", "c"]) == [1, 0, 0]
    # integer set
    assert one_hot_encode(2, [0, 1, 2]) == [0, 0, 1]
    # include_unknown_set is False
    assert one_hot_encode(3, [0, 1, 2]) == [0, 0, 0]
    # include_unknown_set is True
    assert one_hot_encode(3, [0, 1, 2], True) == [0, 0, 0, 1]
