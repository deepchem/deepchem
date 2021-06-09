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
      dc.utils.data_utils.download_url(url=USPTO_MIT_TEST,
                                       dest_dir=self.data_dir)
    loader = dc.data.CSVLoader(tasks=self.tasks,
                               feature_field="smiles",
                               featurizer=self.featurizer)
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

  pass
