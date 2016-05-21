import os
import shutil
import tempfile
import numpy as np
from deepchem.models.tests import TestAPI
from deepchem.utils.save import load_from_disk
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.datasets import Dataset
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.sklearn_models import SklearnModel
from deepchem.featurizers.fingerprints import CircularFingerprint

class TestDrop(TestAPI):
  """
  Test how loading of malformed compounds is handled.

  Called TestDrop since these compounds were silently and erroneously dropped.
  """

  def test_drop(self):
    """Test on dataset where RDKit fails on some strings."""
    # Set some global variables up top
    reload = True
    verbosity = "high"
    len_full = 25

    current_dir = os.path.dirname(os.path.realpath(__file__))
    feature_dir = os.path.join(self.base_dir, "features")
    samples_dir = os.path.join(self.base_dir, "samples")
    full_dir = os.path.join(self.base_dir, "full_dataset")
    model_dir = os.path.join(self.base_dir, "model")

    print("About to load emols dataset.")
    dataset_file = os.path.join(
        current_dir, "mini_emols.csv")

    # Featurize emols dataset
    print("About to featurize datasets.")
    featurizers = [CircularFingerprint(size=1024)]
    emols_tasks = ['activity']

    featurizer = DataFeaturizer(tasks=emols_tasks,
                                smiles_field="smiles",
                                compound_featurizers=featurizers,
                                verbosity=verbosity)
    featurized_samples = featurizer.featurize(
        dataset_file, feature_dir,
        samples_dir, reload=reload)
    print("len(featurized_samples)")
    print(len(featurized_samples))

    # Generate datasets
    dataset = Dataset(data_dir=full_dir, samples=featurized_samples, 
                      featurizers=featurizers, tasks=emols_tasks,
                      verbosity=verbosity, reload=reload)

    X, y, w, ids = dataset.to_numpy()
    print("ids.shape, X.shape, y.shape, w.shape")
    print(ids.shape, X.shape, y.shape, w.shape)
    assert len(X) == len(y) == len(w) == len(ids)
