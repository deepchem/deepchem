import unittest
import tempfile
import numpy as np
import deepchem as dc



class IndiceSplitterTest(unittest.TestCase):


  def test_indice_split(self):

    solubility_dataset = dc.data.tests.load_solubility_data()
    random_splitter = IndiceSplitter(valid_indices=[7],test_indices=[8])
    train_data, valid_data, test_data = \
      random_splitter.split(
        solubility_dataset)
    assert len(train_data) == 8
    assert len(valid_data) == 1
    assert len(test_data) == 1
if __name__ =='__main__':
	unittest.main()