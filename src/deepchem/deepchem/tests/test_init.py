import unittest

import deepchem as dc


class TestInit(unittest.TestCase):

  def test_version(self):
    assert dc.__version__
