"""
MUV dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
import deepchem as dc


def load_permeability(featurizer='ECFP', split='index'):
  """Load membrain permeability datasets. Does not do train/test split"""
  print("About to load membrain permeability dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir,
                              "../../datasets/membrane_permeability.sdf")
  # Featurize permeability dataset
  print("About to featurize membrain permeability dataset.")

  if featurizer == 'ECFP':
    featurizer_func = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer_func = dc.feat.ConvMolFeaturizer()

  permeability_tasks = sorted(['LogP(RRCK)'])

  loader = dc.data.SDFLoader(
      tasks=permeability_tasks, clean_mols=True, featurizer=featurizer_func)
  dataset = loader.featurize(dataset_file)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return permeability_tasks, (train, valid, test), []
