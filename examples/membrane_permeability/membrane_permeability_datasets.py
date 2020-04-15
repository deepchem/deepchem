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
  """Load membrane permeability datasets. Does not do train/test split

  RRCK permeability dataset. This set contains permeability data and structures for 201 compounds curated from 8 sources using the RRCK (aka MDCK-LE) permeability assay. The dataset has been used to train a 3D physics-based permeability model as referenced in http://doi.org/10.1021/acs.jcim.6b00005. Here's the citation for the paper:

  Leung, Siegfried SF, Daniel Sindhikara, and Matthew P. Jacobson. "Simple predictive models of passive membrane permeability incorporating size-dependent membrane-water partition." Journal of chemical information and modeling 56.5 (2016): 924-929.


  Looks like we're using the 3d coordinates here. Just like ESOL, permeability should be based on the compound not conformer, but the conformational ensemble highly affects the permeability (solubility as well). Existing predictors of permeability and solubility both require sampling of the 3d structures.
  """
  print("About to load membrane permeability dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(current_dir,
                              "../../datasets/membrane_permeability.sdf")
  # Featurize permeability dataset
  print("About to featurize membrane permeability dataset.")

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
