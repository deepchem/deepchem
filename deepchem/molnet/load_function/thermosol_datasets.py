"""
Thermodynamic Solubility dataset loader.
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem

logger = logging.getLogger(__name__)


def load_delaney(featurizer='ECFP', split='index', reload=True):
  """Load Thermodynamic Solubility datasets."""
  logger.info("About to featurize thermosol dataset.")
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "thermosol/" + featurizer + "/" + split)

  dataset_file = os.path.join(data_dir, "thermosol.csv")
