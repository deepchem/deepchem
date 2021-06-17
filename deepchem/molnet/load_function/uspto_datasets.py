"""
Loads synthetic reaction datasets from USPTO.

This file contains loaders for synthetic reaction datasets from the US Patent Office. http://nextmovesoftware.com/blog/2014/02/27/unleashing-over-a-million-reactions-into-the-wild/.
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

USPTO_MIT_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv"
USPTO_STEREO_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_STEREO.csv"
USPTO_50K_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv"


class _USPTOLoader(_MolnetLoader):

  def __init__(self, *args, subset: str, sep_reagent: bool, **kwargs):
    super(_USPTOLoader, self).__init__(*args, **kwargs)
    self.subset = subset
    self.sep_reagent = sep_reagent
    self.name = 'USPTO_' + subset

  def create_dataset(self) -> Dataset:
    if self.subset not in ['MIT', 'STEREO', '50K']:
      raise ValueError("Valid Subset names are MIT, STEREO and 50K.")

    if self.subset == 'MIT':
      dataset_url = USPTO_MIT_URL

    if self.subset == 'STEREO':
      dataset_url = USPTO_STEREO_URL

    if self.subset == '50K':
      dataset_url = USPTO_50K_URL

    dataset_file = os.path.join(self.data_dir, self.name + '.csv')

    if not os.path.exists(dataset_file):
      logger.info("Downloading dataset...")
      dc.utils.data_utils.download_url(url=dataset_url, dest_dir=self.data_dir)
      logger.info("Dataset download complete.")

    loader = dc.data.CSVLoader(tasks=[], featurizer=self.featurizer)

    return loader.create_dataset(dataset_file, shard_size=8192)


def load_uspto(
    featurizer=None,  # should I remove this?
    splitter=None,
    transformers=None,
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    subset: str = "MIT",
    sep_reagent: bool = True,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:

  featurizer = dc.feat.UserDefinedFeaturizer([])
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
