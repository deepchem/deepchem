"""
Testing singletask/multitask data loading capabilities.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL"

import os
import shutil
import tempfile
import numpy as np
from deepchem.models.tests import TestAPI
from deepchem.utils.save import load_from_disk
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.datasets import Dataset

## task0: 1,1,0,-,0,-,1,-,-,1

class TestLoad(TestAPI):
  """
  Test singletask/multitask data loading.
  """

  def test_singletask_matches_multitask_load(self):
    """Check that singletask load and multitask load of dataset are same."""
    # Only for debug!
    np.random.seed(123)

    # Set some global variables up top
    reload = True
    verbosity = "high"

    base_dir = tempfile.mkdtemp()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    #Make directories to store the raw and featurized datasets.
    feature_dir = os.path.join(base_dir, "features")
    samples_dir = os.path.join(base_dir, "samples")
    full_dir = os.path.join(base_dir, "full_dataset")
    train_dir = os.path.join(base_dir, "train_dataset")
    valid_dir = os.path.join(base_dir, "valid_dataset")
    test_dir = os.path.join(base_dir, "test_dataset")
    model_dir = os.path.join(base_dir, "model")

    # Load dataset
    print("About to load dataset.")
    dataset_file = os.path.join(
        current_dir, "../../models/tests/multitask_example.csv")
    dataset = load_from_disk(dataset_file)
    print("Columns of dataset: %s" % str(dataset.columns.values))
    print("Number of examples in dataset: %s" % str(dataset.shape[0]))

    # Featurize tox21 dataset
    print("About to featurize dataset.")
    featurizers = [CircularFingerprint(size=1024)]
    all_tasks = ["task%d"%i for i in range(17)] 
    # For debugging purposes
    n_tasks = 17 
    tasks = all_tasks[0:n_tasks]
    valid_scores = {}

    ####### Do multitask load
    if os.path.exists(feature_dir):
      shutil.rmtree(feature_dir)
    featurizer = DataFeaturizer(tasks=tasks,
                                smiles_field="smiles",
                                compound_featurizers=featurizers,
                                verbosity=verbosity)
    featurized_samples = featurizer.featurize(
        dataset_file, feature_dir,
        samples_dir, shard_size=8192,
        reload=reload)
    if os.path.exists(full_dir):
      shutil.rmtree(full_dir)
    full_dataset = Dataset(data_dir=full_dir, samples=featurized_samples, 
                            featurizers=featurizers, tasks=tasks,
                            verbosity=verbosity, reload=reload)

    # Do train/valid split.
    X_multi, y_multi, w_multi, ids_multi = full_dataset.to_numpy()


    ####### Do singletask load
    X_tasks, y_tasks, w_tasks, ids_tasks = [], [], [], []
    for task in tasks:
      print("Processing task %s" % task)
      if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
      featurizer = DataFeaturizer(tasks=[task],
                                  smiles_field="smiles",
                                  compound_featurizers=featurizers,
                                  verbosity=verbosity)
      featurized_samples = featurizer.featurize(
          dataset_file, feature_dir,
          samples_dir, shard_size=8192,
          reload=reload)
      if os.path.exists(full_dir):
        shutil.rmtree(full_dir)
      full_dataset = Dataset(data_dir=full_dir, samples=featurized_samples, 
                              featurizers=featurizers, tasks=[task],
                              verbosity=verbosity, reload=reload)

      X_task, y_task, w_task, ids_task = full_dataset.to_numpy()
      X_tasks.append(X_task)
      y_tasks.append(y_task)
      w_tasks.append(w_task)
      ids_tasks.append(ids_task)

    ################## Do comparison
    for ind, task in enumerate(tasks):
      y_multi_task = y_multi[:, ind]
      w_multi_task = w_multi[:, ind]

      #X_task = X_tasks[ind]
      y_task = y_tasks[ind]
      w_task = w_tasks[ind]
      ids_task = ids_tasks[ind]

      np.testing.assert_allclose(y_multi_task.flatten(), y_task.flatten())
      np.testing.assert_allclose(w_multi_task.flatten(), w_task.flatten())
    shutil.rmtree(base_dir)
