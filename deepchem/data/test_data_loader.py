import os
from unittest import TestCase
from io import StringIO
import tempfile
import shutil

import deepchem as dc


class TestCSVLoader(TestCase):

  def test_load_singleton_csv(self):
    fin = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fin.write("smiles,endpoint\nc1ccccc1,1")
    fin.close()
    print(fin.name)
    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["endpoint"]
    loader = dc.data.CSVLoader(
        tasks=tasks, smiles_field="smiles", featurizer=featurizer)

    X = loader.featurize(fin.name)
    self.assertEqual(1, len(X))
    os.remove(fin.name)
