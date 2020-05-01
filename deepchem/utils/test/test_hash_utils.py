import unittest
import numpy as np
from deepchem.utils import hash_utils


def random_string(length, chars=None):
  import string
  if chars is None:
    chars = list(string.ascii_letters + string.ascii_letters + '()[]+-.=#@/\\')
  return ''.join(np.random.choice(chars, length))


class TestHashUtils(unittest.TestCase):

  def test_hash_ecfp(self):
    for power in (2, 16, 64):
      for _ in range(10):
        string = random_string(10)
        string_hash = hash_utils.hash_ecfp(string, power)
        self.assertIsInstance(string_hash, int)
        self.assertLess(string_hash, 2**power)
        self.assertGreaterEqual(string_hash, 0)

  def test_hash_ecfp_pair(self):
    for power in (2, 16, 64):
      for _ in range(10):
        string1 = random_string(10)
        string2 = random_string(10)
        pair_hash = hash_utils.hash_ecfp_pair((string1, string2), power)
        self.assertIsInstance(pair_hash, int)
        self.assertLess(pair_hash, 2**power)
        self.assertGreaterEqual(pair_hash, 0)

  def test_vectorize(self):
    size = 16
    feature_dict = {0: "C", 1: "CC", 2: "CCC"}
    hash_function = hash_utils.hash_ecfp
    vector = hash_utils.vectorize(hash_function, feature_dict, size)
    assert vector.shape == (size,)
    assert np.count_nonzero(vector) == len(feature_dict)
