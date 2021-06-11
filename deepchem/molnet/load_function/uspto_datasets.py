"""
Loads synthetic reaction datasets from USPTO.

This file contains loaders for synthetic reaction datasets from the US Patenent Office. http://nextmovesoftware.com/blog/2014/02/27/unleashing-over-a-million-reactions-into-the-wild/.
"""
import os
import logging
import deepchem
import numpy as np
from deepchem.data import Dataset
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

  def __init__(self, *args, subset: str, sep_reagent: bool, **kwargs):
    super(_USPTOLoader, self).__init__(*args, **kwargs)
    self.subset = subset
    self.sep_reagent = sep_reagent
    self.name = 'USPTO_' + subset

  def create_dataset(self) -> Tuple[Dataset, ...]:
    #####INCOMPLETE/INCORRECT: I don'd think this is the right way to bypass the splitter!
    if self.subset not in ['MIT', 'STEREO']:
      raise ValueError("Valid Subset names are MIT and STEREO.")

    if self.subset == 'MIT':
      train_file = os.path.join(self.data_dir, USPTO_MIT_TRAIN)
      test_file = os.path.join(self.data_dir, USPTO_MIT_TEST)
      valid_file = os.path.join(self.data_dir, USPTO_MIT_VALID)

      if not os.path.exists(train_file):

        logger.info("Downloading training file...")
        dc.utils.data_utils.download_url(
            url=USPTO_MIT_TRAIN, dest_dir=self.data_dir)
        logger.info("Training file download complete.")

        logger.info("Downloading test file...")
        dc.utils.data_utils.download_url(
            url=USPTO_MIT_TEST, dest_dir=self.data_dir)
        logger.info("Test file download complete.")

        logger.info("Downloading validation file...")
        dc.utils.data_utils.download_url(
            url=USPTO_MIT_VALID, dest_dir=self.data_dir)
        logger.info("Validation file download complete.")
      if self.subset == 'STEREO':
        train_file = os.path.join(self.data_dir, USPTO_STEREO_TRAIN)
        test_file = os.path.join(self.data_dir, USPTO_STEREO_TEST)
        valid_file = os.path.join(self.data_dir, USPTO_STEREO_VALID)

        if not os.path.exists(train_file):

          logger.info("Downloading training file...")
          dc.utils.data_utils.download_url(
              url=USPTO_STEREO_TRAIN, dest_dir=self.data_dir)
          logger.info("Training file download complete.")

          logger.info("Downloading test file...")
          dc.utils.data_utils.download_url(
              url=USPTO_STEREO_TEST, dest_dir=self.data_dir)
          logger.info("Test file download complete.")

          logger.info("Downloading validation file...")
          dc.utils.data_utils.download_url(
              url=USPTO_STEREO_VALID, dest_dir=self.data_dir)
          logger.info("Validation file download complete.")

    loader = dc.data.CSVLoader(
        tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer)

    logger.info("Loading train dataset..")
    train_file = loader.create_dataset(train_file, shard_size=8192)
    logger.info("Loading test dataset..")
    test_file = loader.create_dataset(test_file, shard_size=8192)
    logger.info("Loading validation dataset..")
    valid_file = loader.create_dataset(valid_file, shard_size=8192)
    logger.info("Loading successful!")

    #need to figure out how to return the train, test and valid files!
    return (train_file, test_file, valid_file)  


def load_uspto(
    featurizer=None,
    splitter=None,
    transformers=None,
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    subset: str = "MIT",
    sep_reagent: bool = True,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:

  loader = _USPTOLoader(
      featurizer,
      splitter,
      transformers,
      data_dir,
      save_dir,
      subset=subset,
      sep_reagent=sep_reagent,
      **kwargs)
  return loader.load_dataset(loader.name, reload)
