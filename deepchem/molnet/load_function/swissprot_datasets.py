"""
swissprot dataset loader.
"""
import os
import numpy as np
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

SWISSP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/uniprot_swissprot_14_06_21.csv"
SWISSP_TASK = ["0"]


class _SWISSPROTLoader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, "uniprot_swissprot_14_06_21.csv")
    print(dataset_file)
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=SWISSP_URL, dest_dir=self.data_dir)
      dataset_file = os.path.join(self.data_dir,
                                  "uniprot_swissprot_14_06_21.csv")
    loader = dc.data.CSVLoader(
        tasks=self.tasks,
        featurizer=self.featurizer,
        feature_field="SEQUENCE",
    )
    return loader.create_dataset(dataset_file)  #,shard_size=8192)


def load_swissprot(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.OneHotFeaturizer([
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
        'R', 'S', 'T', 'V', 'W', 'Y'
    ]),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs) -> Tuple[List[str], Tuple[Dataset, ...]]:
    
  """
    Load Swissprot dataset.

    The Swiss-Prot Database is part of the UniprotKnowledebase (UniprotKB) that contains a
    collection of functionañ information on protiens. Specially the Swiss-Prot Database 
    contains records with information extracted from literature and curator-evaluated computational
    analysis. 
    https://www.uniprot.org/uniprot/?query=reviewed:yes
    Parameters
    ----------
    featurizer: Featurizer or str
      the featurizer to use for processing the data.  Alternatively you can pass
      one of the names from dc.molnet.featurizers as a shortcut.
    splitter: Splitter or str
      the splitter to use for splitting the data into training, validation, and
      test sets.  Alternatively you can pass one of the names from
      dc.molnet.splitters as a shortcut.  If this is None, all the data
      will be included in a single dataset.
    reload: bool
      if True, the first call for a particular featurizer and splitter will cache
      the datasets to disk, and subsequent calls will reload the cached datasets.
    data_dir: str
      a directory to save the raw data in
    save_dir: str
      a directory to save the dataset in

    Note
    ----
    The uniportKB update the swissprot data base each 8 weeks. For download the whole uniprotKB
    please visit : https://www.uniprot.org/downloads
    The version avaible in the AWS Bucket is from 14/06/21


    This version of the swissprot DB contains 564638 curated sequences. The dataset contains a 
    featurized sequences ( Using the one hot featurizer by default) 
    

    References
    ----------
    .. [1] UniProt Consortium, T. UniProt: The Universal Protein Knowledgebase.
     Nucleic Acids Res 2018, 46 (5), 2699–2699. https://doi.org/10.1093/nar/gky092.

    """

  loader = _SWISSPROTLoader(featurizer, splitter, transformers, SWISSP_TASK,
                            data_dir, save_dir, **kwargs)
  return loader.load_dataset('swissprot', reload)
