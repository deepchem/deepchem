"""
Common code for loading MoleculeNet datasets.
"""
import os
import logging
import deepchem as dc
from deepchem.data import Dataset, DiskDataset
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

featurizers = {
    'ecfp': dc.feat.CircularFingerprint(size=1024),
    'graphconv': dc.feat.ConvMolFeaturizer(),
    'weave': dc.feat.WeaveFeaturizer(),
    'raw': dc.feat.RawFeaturizer(),
    'smiles2img': dc.feat.SmilesToImage(img_size=80, img_spec='std')
}

splitters = {
    'index': dc.splits.IndexSplitter(),
    'random': dc.splits.RandomSplitter(),
    'scaffold': dc.splits.ScaffoldSplitter(),
    'butina': dc.splits.ButinaSplitter(),
    'task': dc.splits.TaskSplitter(),
    'stratified': dc.splits.RandomStratifiedSplitter()
}


class _MolnetLoader(object):
  """The class provides common functionality used by many molnet loader functions.
  It is an abstract class.  Subclasses implement loading of particular datasets.
  """

  def __init__(self, featurizer: Union[dc.feat.Featurizer, str],
               splitter: Union[dc.splits.Splitter, str, None],
               data_dir: Optional[str], save_dir: Optional[str], **kwargs):
    """Construct an object for loading a dataset.

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
    data_dir: str
      a directory to save the raw data in
    save_dir: str
      a directory to save the dataset in
    """
    if 'split' in kwargs:
      splitter = kwargs['split']
      logger.warning("'split' is deprecated.  Use 'splitter' instead.")
    if isinstance(featurizer, str):
      featurizer = featurizers[featurizer.lower()]
    if isinstance(splitter, str):
      splitter = splitters[splitter.lower()]
    if data_dir is None:
      data_dir = dc.utils.data_utils.get_data_dir()
    if save_dir is None:
      save_dir = dc.utils.data_utils.get_data_dir()
    self.featurizer = featurizer
    self.splitter = splitter
    self.data_dir = data_dir
    self.save_dir = save_dir
    self.args = kwargs

  def load_dataset(
      self, tasks: List[str], save_folder: str, reload: bool
  ) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load the dataset.

    Parameters
    ----------
    tasks: List[str]
      the names of the tasks in this dataset
    save_folder: str
      the directory in which the dataset should be saved
    reload: bool
      if True, the first call for a particular featurizer and splitter will cache
      the datasets to disk, and subsequent calls will reload the cached datasets.
    """
    # Try to reload cached datasets.

    if reload:
      if self.splitter is None:
        if os.path.exists(save_folder):
          transformers = dc.utils.data_utils.load_transformers(save_folder)
          return tasks, (DiskDataset(save_folder),), transformers
      else:
        loaded, all_dataset, transformers = dc.utils.data_utils.load_dataset_from_disk(
            save_folder)
        if all_dataset is not None:
          return tasks, all_dataset, transformers

    # Create the dataset

    dataset = self.create_dataset()

    # Split and transform the dataset.

    if self.splitter is None:
      transformer_dataset: Dataset = dataset
    else:
      logger.info("About to split dataset with {} splitter.".format(
          self.splitter.__class__.__name__))
      train, valid, test = self.splitter.train_valid_test_split(dataset)
      transformer_dataset = train
    transformers = self.get_transformers(transformer_dataset)
    logger.info("About to transform data.")
    if self.splitter is None:
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

  def create_dataset(self) -> Dataset:
    """Subclasses must implement this to load the dataset."""
    raise NotImplementedError()

  def get_transformers(self, dataset: Dataset) -> List[dc.trans.Transformer]:
    """Subclasses must implement this to create the transformers for the dataset."""
    raise NotImplementedError()
