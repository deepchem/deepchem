import unittest


class TestDeepchemBuild(unittest.TestCase):
  def setUp(self):
    pass

  def test_dc_import(self):
    import deepchem
    print(deepchem.__version__)

  def test_rdkit_import(self):
    import rdkit
    print(rdkit.__version__)

  def test_numpy_import(self):
    import numpy as np
    print(np.__version__)

  def test_pandas_import(self):
    import pandas as pd
    print(pd.__version__)

if __name__ == '__main__':
  unittest.main()
