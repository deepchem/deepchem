"""
Delaney dataset loader.
"""
import os
import logging
import deepchem as dc
from deepchem.data import Dataset, DiskDataset
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_DIR = dc.utils.data_utils.get_data_dir()
DELANEY_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"


def load_delaney(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    reload: bool = True,
    move_mean: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Load delaney dataset

  The Delaney(ESOL) dataset a regression dataset containing structures and
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
    one of the names from dc.molnet.defaults.featurizers as a shortcut.
  splitter: Splitter or str
    the splitter to use for splitting the data into training, validation, and
    test sets.  Alternatively you can pass one of the names from
    dc.molnet.defaults.splitters as a shortcut.  If this is None, all the data
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
  if 'split' in kwargs:
    splitter = kwargs['split']
    logger.warning("'split' is deprecated.  Use 'splitter' instead.")
  if isinstance(featurizer, str):
    featurizer = dc.molnet.defaults.featurizers[featurizer]
  if isinstance(splitter, str):
    splitter = dc.molnet.defaults.splitters[splitter]
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR
  tasks = ['measured log solubility in mols per litre']

  # Try to reload cached datasets.

  if reload:
    featurizer_name = str(featurizer.__class__.__name__)
    splitter_name = str(splitter.__class__.__name__)
    if not move_mean:
      featurizer_name = featurizer_name + "_mean_unmoved"
    save_folder = os.path.join(save_dir, "delaney-featurized", featurizer_name,
                               splitter_name)
    if splitter is None:
      if os.path.exists(save_folder):
        transformers = dc.utils.data_utils.load_transformers(save_folder)
        return tasks, (DiskDataset(save_folder),), transformers
    else:
      loaded, all_dataset, transformers = dc.utils.data_utils.load_dataset_from_disk(
          save_folder)
      if all_dataset is not None:
        return tasks, all_dataset, transformers

  # Featurize Delaney dataset

  logger.info("About to featurize Delaney dataset.")
  dataset_file = os.path.join(data_dir, "delaney-processed.csv")
  if not os.path.exists(dataset_file):
    dc.utils.data_utils.download_url(url=DELANEY_URL, dest_dir=data_dir)
  loader = dc.data.CSVLoader(
      tasks=tasks, feature_field="smiles", featurizer=featurizer)
  dataset = loader.create_dataset(dataset_file, shard_size=8192)

  # Split and transform the dataset.

  if splitter is None:
    transformer_dataset: Dataset = dataset
  else:
    logger.info("About to split dataset with {} splitter.".format(
        splitter.__class__.__name__))
    train, valid, test = splitter.train_valid_test_split(dataset)
    transformer_dataset = train
  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=transformer_dataset, move_mean=move_mean)
  ]
  logger.info("About to transform data.")
  if splitter is None:
    for transformer in transformers:
      dataset = transformer.transform(dataset)
    if reload and isinstance(dataset, DiskDataset):
      dataset.move(save_folder)
      dc.utils.data_utils.save_transformers(save_folder, transformers)
    return tasks, (dataset,), transformers

  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)
  if reload and isinstance(train, DiskDataset) and isinstance(
      valid, DiskDataset) and isinstance(test, DiskDataset):
    dc.utils.data_utils.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return tasks, (train, valid, test), transformers
