"""
Testing reload.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from deepchem.models.test import TestAPI
from deepchem.utils.save import load_from_disk
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.splits import ScaffoldSplitter
from deepchem.datasets import Dataset
from deepchem.transformers import BalancingTransformer
from deepchem.hyperparameters import HyperparamOpt
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel

class TestReload(TestAPI):
  """
  Test reload for datasets.
  """

  def _run_muv_experiment(self, reload=False, verbosity=None):
    """Loads or reloads a small version of MUV dataset."""
    # Load MUV dataset
    dataset_file = os.path.join(
        self.current_dir, "../../../datasets/mini_muv.csv.gz")
    dataset = load_from_disk(dataset_file)
    print("Number of examples in dataset: %s" % str(dataset.shape[0]))

    featurizers = [CircularFingerprint(size=1024)]
    MUV_tasks = ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644',
                 'MUV-548', 'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712',
                 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733', 'MUV-652',
                 'MUV-466', 'MUV-832']
    featurizer = DataFeaturizer(tasks=MUV_tasks,
                                smiles_field="smiles",
                                compound_featurizers=featurizers,
                                verbosity="low")
    featurized_samples = featurizer.featurize(
        dataset_file, self.feature_dir,
        self.samples_dir, shard_size=4096,
        reload=reload)

    splitter = ScaffoldSplitter()
    train_samples, valid_samples, test_samples = \
        splitter.train_valid_test_split(
            featurized_samples, self.train_dir, self.valid_dir, self.test_dir,
            log_every_n=1000, reload=reload)
    len_train_samples, len_valid_samples, len_test_samples = \
      len(train_samples), len(valid_samples), len(test_samples)

    print("Creating train dataset.")
    train_dataset = Dataset(data_dir=self.train_dir, samples=train_samples, 
                            featurizers=featurizers, tasks=MUV_tasks,
                            verbosity=verbosity, reload=reload)
    print("Creating valid dataset.")
    valid_dataset = Dataset(data_dir=self.valid_dir, samples=valid_samples, 
                            featurizers=featurizers, tasks=MUV_tasks,
                            verbosity=verbosity, reload=reload)
    print("Creating test dataset")
    test_dataset = Dataset(data_dir=self.test_dir, samples=test_samples, 
                           featurizers=featurizers, tasks=MUV_tasks,
                           verbosity=verbosity, reload=reload)
    len_train_dataset, len_valid_dataset, len_test_dataset = \
      len(train_dataset), len(valid_dataset), len(test_dataset)

    input_transformers = []
    output_transformers = []
    weight_transformers = [BalancingTransformer(transform_w=True,
    dataset=train_dataset)]
    transformers = input_transformers + output_transformers + weight_transformers
    print("Transforming train dataset")
    for transformer in transformers:
        transformer.transform(train_dataset)
    print("Transforming valid dataset")
    for transformer in transformers:
        transformer.transform(valid_dataset)
    print("Transforming test dataset")
    for transformer in transformers:
        transformer.transform(test_dataset)

    return (len_train_samples, len_valid_samples, len_test_samples,
            len_train_dataset, len_valid_dataset, len_test_dataset)
    
  def test_reload(self):
    """Check num samples for loaded and reload datasets is equal."""
    reload = False
    verbosity = None
    print("Running experiment for first time w/o reload.")
    (len_train_samples, len_valid_samples, len_test_samples,
     len_train_dataset, len_valid_dataset, len_test_dataset) = \
        self._run_muv_experiment(reload, verbosity)

    print("Running experiment for second time with reload.")
    reload = True 
    (len_reload_train_samples, len_reload_valid_samples, len_reload_test_samples,
     len_reload_train_dataset, len_reload_valid_dataset, len_reload_test_dataset) = \
        self._run_muv_experiment(reload, verbosity)
    assert len_train_samples == len_reload_train_samples
    assert len_valid_samples == len_reload_valid_samples
    assert len_test_samples == len_reload_valid_samples
    assert len_train_dataset == len_reload_train_dataset
    assert len_valid_dataset == len_reload_valid_dataset
    assert len_test_dataset == len_reload_valid_dataset
