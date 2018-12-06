"""
Loads synthetic reaction datasets from USPTO.

This file contains loaders for synthetic reaction datasets from the US Patenent Office. http://nextmovesoftware.com/blog/2014/02/27/unleashing-over-a-million-reactions-into-the-wild/.
"""

from __future__ import division
from __future__ import unicode_literals

import os
import csv
import logging
import deepchem
import numpy as np
from deepchem.data import DiskDataset

logger = logging.getLogger(__name__)


def load_uspto(featurizer="plain",
               split=None,
               num_to_load=10000,
               reload=True,
               verbose=False):
  """Load USPTO dataset.

  For now, only loads the subset of data for 2008-2011 reactions. See https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873 for more details.

  The full dataset contains some 400K reactions. This causes an out-of-memory error on development laptop if full dataset is featurized. For now, return a truncated subset of dataset. 

  Reloading is not entirely supported for this dataset.
  """
  # Most reaction dataset ML tasks train the prediction of products from
  # ractants. Both of these are contained in the rxn object that is output,
  # so there is no "tasks" field.
  uspto_tasks = []
  # DeepChem currently has no transformers for reaction data
  uspto_transformers = []
  if split is not None:
    raise ValueError("Train/valid/test not yet supported.")
  # Download USPTO dataset
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "uspto/" + featurizer + "/")
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return uspto_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir,
                              "2008-2011_USPTO_reactionSmiles_filtered.zip")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(
        "https://bitbucket.org/dan2097/patent-reaction-extraction/downloads/2008-2011_USPTO_reactionSmiles_filtered.zip"
    )
  # Unzip
  unzip_dir = os.path.join(data_dir, "2008-2011_USPTO_reactionSmiles_filtered")
  if not os.path.exists(unzip_dir):
    deepchem.utils.unzip_file(dataset_file, dest_dir=unzip_dir)
  # Unzipped file is a tap seperated values file (despite the .txt)
  filename = os.path.join(unzip_dir,
                          "2008-2011_USPTO_reactionSmiles_filtered.txt")
  rxns = []
  from rdkit.Chem import rdChemReactions
  with open(filename) as tsvfile:
    reader = csv.reader(tsvfile, delimiter="\t")
    for ind, row in enumerate(reader):
      if ind > num_to_load:
        break
      if verbose:
        print("Loading reaction %d" % ind)
      # The first element in the row is the reaction smarts
      smarts = row[0]
      # Sometimes smarts have extraneous information at end of form "
      # |f:0" that causes parsing to fail. Not sure what this information
      # is, but just ignoring for now.
      smarts = smarts.split(" ")[0]
      rxn = rdChemReactions.ReactionFromSmarts(smarts)
      rxns.append(rxn)
  rxn_array = np.array(rxns)
  # Make up dummy labels since DiskDataset.from_numpy doesn't allow
  # creation from just features for now.
  y = np.ones(len(rxn_array))
  # TODO: This dataset isn't saved to disk so reload doesn't happen.
  rxn_dataset = DiskDataset.from_numpy(rxn_array, y)
  return uspto_tasks, (rxn_dataset, None, None), uspto_transformers
