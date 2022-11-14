from deepchem.utils.jackhmmer import Jackhmmer
import unittest

class TestJackhmmer(unittest.TestCase):
  """
  Test Jackhmmer  on a toy dataset
  """
  def test_jackhmmer(self):
    j = Jackhmmer(database_path='assets/test.fasta')
    result = j.query("assets/sequence.fasta")
    exp_val = '#=GC RF xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    for line in result[0]['sto'].split('\n'):
      if line.startswith('#=GC'):
        self.assertEqual(line,exp_val)