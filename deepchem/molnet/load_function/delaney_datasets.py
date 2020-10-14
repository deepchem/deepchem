"""
Delaney dataset loader.
"""
import os
import logging
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DELANEY_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
DELANEY_TASKS = ['measured log solubility in mols per litre']


class _DelaneyLoader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    logger.info("About to featurize Delaney dataset.")
    dataset_file = os.path.join(self.data_dir, "delaney-processed.csv")
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=DELANEY_URL, dest_dir=self.data_dir)
    loader = dc.data.CSVLoader(
        tasks=DELANEY_TASKS, feature_field="smiles", featurizer=self.featurizer)
    return loader.create_dataset(dataset_file, shard_size=8192)

  def get_transformers(self, dataset: Dataset) -> List[dc.trans.Transformer]:
    return [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=self.args['move_mean'])
    ]


def load_delaney(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    reload: bool = True,
    move_mean: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Load Delaney dataset

  The Delaney (ESOL) dataset a regression dataset containing structures and
  water solubility data for 1128 compounds. The dataset is widely used to
  validate machine learning models on estimating solubility directly from
  molecular structures (as encoded in SMILES strings).

  Scaffold splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "Compound ID" - Name of the compound
  - "smiles" - SMILES representation of the molecular structure
  - "measured log solubility in mols per litre" - Log-scale water solubility
    of the compound, used as label

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
  move_mean: bool
    if True, all the data is shifted so the training set has a mean of zero.
  data_dir: str
    a directory to save the raw data in
  save_dir: str
    a directory to save the dataset in

  References
  ----------
  .. [1] Delaney, John S. "ESOL: estimating aqueous solubility directly from
     molecular structure." Journal of chemical information and computer
     sciences 44.3 (2004): 1000-1005.
  """
  loader = _DelaneyLoader(
      featurizer, splitter, data_dir, save_dir, move_mean=move_mean, **kwargs)
  featurizer_name = str(loader.featurizer)
  splitter_name = 'None' if loader.splitter is None else str(loader.splitter)
  if not move_mean:
    featurizer_name = featurizer_name + "_mean_unmoved"
  save_folder = os.path.join(loader.save_dir, "delaney-featurized",
                             featurizer_name, splitter_name)
  return loader.load_dataset(DELANEY_TASKS, save_folder, reload)
