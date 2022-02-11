import unittest
import os
import platform

from deepchem.utils.sequence_utils import hhblits, hhsearch

IS_WINDOWS = platform.system() == 'Windows'

@unittest.skipIf(IS_WINDOWS, "Skip test on Windows")
class TestSeq(unittest.TestCase):
  """
  Tests sequence handling utilities.
  """
  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.dataset_file = os.path.join(current_dir, '../data/example.fasta')
    self.database_file = os.path.join(current_dir,'../') #create test database
    self.data_dir = os.path.join(current_dir,'../data/')
    self.save_dir = os.path.join(current_dir,'../data/')

  def test_load_dataset(self): #Probably not necessary right?
    assert os.path.exists(self.dataset_file)

  def test_load_database(self):
    assert os.path.exists(self.database_file)

  def test_hhsearch(self):
    hhsearch(self.dataset_file, self.database_file)
    assert os.path.exists(self.save_dir)

  def test_hhblits(self):
    hhblits(self.dataset_file, self.database_file)
    assert os.path.exists(self.save_dir)