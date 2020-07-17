"""
Loads synthetic reaction datasets from USPTO.

This file contains loaders for synthetic reaction datasets from the US Patenent Office. http://nextmovesoftware.com/blog/2014/02/27/unleashing-over-a-million-reactions-into-the-wild/.
"""
import os
import csv
import logging
import deepchem
import numpy as np
from deepchem.data import DiskDataset

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
USPTO_URL = "https://bitbucket.org/dan2097/patent-reaction-extraction/downloads/2008-2011_USPTO_reactionSmiles_filtered.zip"

def load_uspto(featurizer="plain",
               split=None,
               num_to_load=10000,
               reload=True,
               data_dir=None,
               save_dir=None,
               **kwargs):
  """Load USPTO dataset.

  For now, only loads the subset of data for 2008-2011 reactions.
  See https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
  for more details. The full dataset contains some 400K reactions. This causes
  an out-of-memory error on development laptop if full dataset is featurized.
  For now, return a truncated subset of dataset.
  Reloading is not entirely supported for this dataset.
  """
  # Most reaction dataset ML tasks train the prediction of
  # products from reactants. Both of these are contained in the
  # rxn object that is output, so there is no "tasks" field.
  uspto_tasks = []

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if split is not None:
    raise ValueError("Train/valid/test not yet supported.")
    
  # Download USPTO dataset
  if reload:
    save_folder = os.path.join(save_dir, "uspto-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return uspto_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir,
                              "2008-2011_USPTO_reactionSmiles_filtered.zip")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=USPTO_URL, dest_dir=data_dir)
  
  if featurizer == 'Raw' or featurizer == 'Plain':
    featurizer = deepchem.feat.RawReactionFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer)
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
      logger.info("Loading reaction %d" % ind)
      # The first element in the row is the reaction smarts
      smarts = row[0]
      # Sometimes smarts have extraneous information at end of
      # form " |f:0" that causes parsing to fail. Not sure what
      # this information is, but just ignoring for now.
      smarts = smarts.split(" ")[0]
      rxn = rdChemReactions.ReactionFromSmarts(smarts)
      rxns.append(rxn)
  rxn_array = np.array(rxns)
  # Make up dummy labels since DiskDataset.from_numpy doesn't allow
  # creation from just features for now.
  y = np.ones(len(rxn_array))
  # TODO: This dataset isn't saved to disk so reload doesn't happen.
  rxn_dataset = DiskDataset.from_numpy(rxn_array, y)
  transformers = []
  return uspto_tasks, (rxn_dataset, None, None), transformers


def load_uspto_old(featurizer="plain",
               split=None,
               num_to_load=10000,
               reload=True,
               data_dir=None,
               save_dir=None,
               **kwargs):
  """Load USPTO dataset.

  For now, only loads the subset of data for 2008-2011 reactions.
  See https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
  for more details. The full dataset contains some 400K reactions. This causes
  an out-of-memory error on development laptop if full dataset is featurized.
  For now, return a truncated subset of dataset.
  Reloading is not entirely supported for this dataset.
  """
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  # Most reaction dataset ML tasks train the prediction of
  # products from reactants. Both of these are contained in the
  # rxn object that is output, so there is no "tasks" field.
  uspto_tasks = []
  if split is not None:
    raise ValueError("Train/valid/test not yet supported.")
  # Download USPTO dataset
  if reload:
    save_folder = os.path.join(save_dir, "uspto-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return uspto_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir,
                              "2008-2011_USPTO_reactionSmiles_filtered.zip")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=USPTO_URL, dest_dir=data_dir)

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
      logger.info("Loading reaction %d" % ind)
      # The first element in the row is the reaction smarts
      smarts = row[0]
      # Sometimes smarts have extraneous information at end of
      # form " |f:0" that causes parsing to fail. Not sure what
      # this information is, but just ignoring for now.
      smarts = smarts.split(" ")[0]
      rxn = rdChemReactions.ReactionFromSmarts(smarts)
      rxns.append(rxn)
  rxn_array = np.array(rxns)
  # Make up dummy labels since DiskDataset.from_numpy doesn't allow
  # creation from just features for now.
  y = np.ones(len(rxn_array))
  # TODO: This dataset isn't saved to disk so reload doesn't happen.
  rxn_dataset = DiskDataset.from_numpy(rxn_array, y)
  transformers = []
  return uspto_tasks, (rxn_dataset, None, None), transformers
