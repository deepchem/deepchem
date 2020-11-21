"""
bace dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union
from deepchem.molnet.load_function.bace_features import bace_user_specified_features

BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
BACE_REGRESSION_TASKS = ["pIC50"]
BACE_CLASSIFICATION_TASKS = ["Class"]


class _BaceLoader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, "bace.csv")
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=BACE_URL, dest_dir=self.data_dir)
    loader = dc.data.CSVLoader(
        tasks=self.tasks, feature_field="mol", featurizer=self.featurizer)
    return loader.create_dataset(dataset_file, shard_size=8192)


def load_bace_regression(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """ Load BACE dataset, regression labels

  The BACE dataset provides quantitative IC50 and qualitative (binary label)
  binding results for a set of inhibitors of human beta-secretase 1 (BACE-1).

  All data are experimental values reported in scientific literature over the
  past decade, some with detailed crystal structures available. A collection
  of 1522 compounds is provided, along with the regression labels of IC50.

  Scaffold splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "mol" - SMILES representation of the molecular structure
  - "pIC50" - Negative log of the IC50 binding affinity
  - "class" - Binary labels for inhibitor

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
  transformers: list of TransformerGenerators or strings
    the Transformers to apply to the data.  Each one is specified by a
    TransformerGenerator or, as a shortcut, one of the names from
    dc.molnet.transformers.
  reload: bool
    if True, the first call for a particular featurizer and splitter will cache
    the datasets to disk, and subsequent calls will reload the cached datasets.
  data_dir: str
    a directory to save the raw data in
  save_dir: str
    a directory to save the dataset in

  References
  ----------
  .. [1] Subramanian, Govindan, et al. "Computational modeling of β-secretase 1
     (BACE-1) inhibitors using ligand based approaches." Journal of chemical
     information and modeling 56.10 (2016): 1936-1949.
  """
  loader = _BaceLoader(featurizer, splitter, transformers,
                       BACE_REGRESSION_TASKS, data_dir, save_dir, **kwargs)
  return loader.load_dataset('bace_r', reload)


def load_bace_classification(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """ Load BACE dataset, classification labels

  BACE dataset with classification labels ("class").

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
  transformers: list of TransformerGenerators or strings
    the Transformers to apply to the data.  Each one is specified by a
    TransformerGenerator or, as a shortcut, one of the names from
    dc.molnet.transformers.
  reload: bool
    if True, the first call for a particular featurizer and splitter will cache
    the datasets to disk, and subsequent calls will reload the cached datasets.
  data_dir: str
    a directory to save the raw data in
  save_dir: str
    a directory to save the dataset in
  """
  loader = _BaceLoader(featurizer, splitter, transformers,
                       BACE_CLASSIFICATION_TASKS, data_dir, save_dir, **kwargs)
  return loader.load_dataset('bace_c', reload)
