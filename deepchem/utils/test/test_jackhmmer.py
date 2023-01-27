from deepchem.utils.jackhmmer import Jackhmmer
import unittest
import pytest


@pytest.mark.fold
def test_jackhmmer():
    j = Jackhmmer(database_path='./assets/test.fasta')
    result = j.query("./assets/sequence.fasta")
    exp_val = '#=GC RF xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    for line in result[0]['sto'].split('\n'):
        if line.startswith('#=GC'):
            assert line == exp_val
            
test_jackhmmer()