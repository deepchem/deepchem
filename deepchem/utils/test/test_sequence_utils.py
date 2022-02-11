import unittest
import os
import platform
from deepchem.utils import sequence_utils as seq_utils

IS_WINDOWS = platform.system() == 'Windows'

@unittest.skipIf(IS_WINDOWS, "Skip test on Windows") #hhsuite does not run on windows
class TestSeq(unittest.TestCase):
  """
  Tests sequence handling utilities.
  """
  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.dataset_file = os.path.join(current_dir, '../data/example.fasta')
    self.database_file = os.path.join(current_dir,'../') #need to create small test database
    self.data_dir = os.path.join(current_dir,'../data/')
    self.save_dir = os.path.join(current_dir,'../data/')

  def test_hhsearch(self):
    seq_utils.hhsearch(self.dataset_file, database = self.database_file)
    f = open('results.a3m', 'r')
    # with open('results.a3m', 'r') as f:
    lines = f.readlines()
    f.close()
    assert len(lines) > 0 # and expected results

  def test_hhblits(self):
    seq_utils.hhblits(self.dataset_file, database = self.database_file)
    f = open('results.a3m', 'r')
    lines = f.readlines()
    f.close()
    assert len(lines) > 0 # and expected results

  