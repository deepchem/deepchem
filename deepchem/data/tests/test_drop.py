import os
import shutil
import unittest
import tempfile
import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class TestDrop(unittest.TestCase):
  """
  Test how loading of malformed compounds is handled.

  Called TestDrop since these compounds were silently and erroneously dropped.
  """

  def test_drop(self):
    """Test on dataset where RDKit fails on some strings."""
    # Set some global variables up top
    reload = True
    len_full = 25

    current_dir = os.path.dirname(os.path.realpath(__file__))
    print("About to load emols dataset.")
    dataset_file = os.path.join(current_dir, "mini_emols.csv")

    # Featurize emols dataset
    print("About to featurize datasets.")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    emols_tasks = ['activity']

    loader = dc.data.CSVLoader(
        tasks=emols_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file)

    X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)
    assert len(X) == len(y) == len(w) == len(ids)
