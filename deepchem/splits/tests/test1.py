import unittest
import tempfile
import numpy as np
import deepchem as dc

class IndiceSplitter():
  """
    Class for splits based on input order.
    """

  def __init__(self, verbose=False, valid_indices=None, test_indices=None):
    """
        Parameters
        -----------
        valid_indices: list of int
            indices of samples in the valid set
        test_indices: list of int
            indices of samples in the test set
        """
    self.verbose = verbose
    self.valid_indices = valid_indices
    self.test_indices = test_indices

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds into train/validation/test in designated order.
        """
    num_datapoints = len(dataset)
    indices = np.arange(num_datapoints).tolist()
    train_indices = []
    if self.valid_indices is None:
      self.valid_indices = []
    if self.test_indices is None:
      self.test_indices = []
    valid_test = list(self.valid_indices)
    valid_test.extend(self.test_indices)
    for indice in indices:
      if not indice in valid_test:
        train_indices.append(indice)

    return (train_indices, self.valid_indices, self.test_indices)


class IndiceSplitterTest(unittest.TestCase):


  def test_indice_split(self):

    solubility_dataset = dc.data.tests.load_solubility_data()
    random_splitter = IndiceSplitter(test_indices=[8])
    train_data, valid_data, test_data = \
      random_splitter.split(
        solubility_dataset)
    assert len(train_data) == 9
    assert len(valid_data) == 0
    assert len(test_data) == 1
if __name__ =='__main__':
	unittest.main()