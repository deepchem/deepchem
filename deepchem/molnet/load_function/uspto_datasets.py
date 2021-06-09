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
from deepchem.molnet.load_function.molnet_loader import _MolnetLoader
from typing import List, Optional, Tuple, Union
import deepchem as dc

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.data_utils.get_data_dir()

USPTO_MIT_TRAIN = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_MIT_train.csv"
USPTO_MIT_TEST = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_MIT_test.csv"
USPTO_MIT_VALID = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_MIT_val.csv"

USPTO_STEREO_TRAIN = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_STEREO_train.csv"
USPTO_STEREO_TEST = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_STEREO_test.csv"
USPTO_STEREO_VALID = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/USPTO_STEREO_val.csv"


class _USPTOLoader(_MolnetLoader):

  def create_dataset(self) -> DiskDataset:
    dataset_file = os.path.join(self.data_dir, "USPTO_MIT_test.csv")
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=USPTO_MIT_TEST, dest_dir=self.data_dir)
    loader = dc.data.CSVLoader(
      tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer)
    return loader.create_dataset(dataset_file, shard_size=8192)


def load_uspto(
    featurizer: Union[dc.feat.Featurizer, str] = None,
    splitter: Union[dc.splits.Splitter, str, None] = None,
    transformers: List[Union[TransformerGenerator, str]] = None,
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[DiskDataset, ...], List[dc.trans.Transformer]]:














"""
def load_uspto(featurizer="plain", #what does the plain featurizer mean? what are the other featurizers?
               split=None,
               num_to_load=10000, # load the whole thing!
               reload=True, #what is reload?
               verbose=False, #what is verbose?
               data_dir=None, #ig this is okay
               save_dir=None,
               **kwargs): ##have to give option to load a particular subset. and option to separate reagents
  """
  """Load USPTO dataset.

  The USPTO Dataset consists of over a million reactions from United States
  patent applications(2001-2013) and the same again from patent grants
  (1976-2013). The loader can load the entire dataset or subsets of it.
  The subsets are USPTO_STEREO, USPTO_MIT and USPTO_50k. The STEREO dataset
  contains around a million reactions with stereochemical information,
  the 50k dataset consists of 50k reactions classified into 10 classes and
  the MIT dataset consists of around 470k reactions. 
  
  https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
  for more details. The full dataset contains some 400K reactions. This causes
  an out-of-memory error on development laptop if full dataset is featurized.
  For now, return a truncated subset of dataset.
  Reloading is not entirely supported for this dataset.
  """
  """
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  # Most reaction dataset ML tasks train the prediction of products from
  # reactants. Both of these are contained in the rxn object that is output,
  # so there is no "tasks" field.
  uspto_tasks = [] #check what is tasks, but im pretty sure there are none!
  if split is not None: ##have to change this, we have train/test/valid ready.
    raise ValueError("Train/valid/test not yet supported.")
  # Download USPTO dataset
  if reload:
    save_folder = os.path.join(save_dir, "uspto-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.data_utils.load_dataset_from_disk(
        save_folder)
    if loaded:
      return uspto_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir,
                              "2008-2011_USPTO_reactionSmiles_filtered.zip")
  if not os.path.exists(dataset_file): # no need to download? since its on AWS
    deepchem.utils.data_utils.download_url(url=USPTO_URL, dest_dir=data_dir)

  # Unzip #NO need to unzip, since its already in .csv
  unzip_dir = os.path.join(data_dir, "2008-2011_USPTO_reactionSmiles_filtered")
  if not os.path.exists(unzip_dir):
    deepchem.utils.data_utils.unzip_file(dataset_file, dest_dir=unzip_dir)
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
      # |f:0" that causes parsing to fail. Not sure what this information, ##yup i need to figure out what that |f thing means as well!
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
  transformers = [] #what are these transformers?
  return uspto_tasks, (rxn_dataset, None, None), transformers #it returns a diskdataset.
"""