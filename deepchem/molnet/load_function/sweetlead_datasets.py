"""
SWEET dataset loader.
"""
import os
import numpy as np
import shutil
import logging
import deepchem as dc

logger = logging.getLogger(__name__)

DEFAULT_DIR = dc.utils.get_data_dir()
SWEETLEAD_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sweet.csv.gz"


def load_sweet(featurizer='ECFP',
               split='index',
               reload=True,
               frac_train=.8,
               data_dir=None,
               save_dir=None,
               **kwargs):
  """Load sweet datasets.

  Sweetlead is a dataset of chemical structures for approved drugs, chemical isolates from traditional medicinal herbs, and regulated chemicals. Resulting structures are filtered for the active pharmaceutical ingredient, standardized, and differing formulations of the same drug were combined in the final database.

  Novick, Paul A., et al. "SWEETLEAD: an in silico database of approved drugs, regulated chemicals, and herbal isolates for computer-aided drug discovery." PLoS One 8.11 (2013).
  """
  # Load Sweetlead dataset
  logger.info("About to load Sweetlead dataset.")
  SWEET_tasks = ["task"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "sweet-featurized", featurizer)
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = dc.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return SWEET_tasks, all_dataset, transformers

  # Featurize SWEET dataset
  logger.info("About to featurize SWEET dataset.")
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == "smiles2img":
    img_spec = kwargs.get("img_spec", "std")
    img_size = kwargs.get("img_size", 80)
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec)
  else:
    raise ValueError("Other featurizations not supported")

  dataset_file = os.path.join(data_dir, "sweet.csv.gz")
  if not os.path.exists(dataset_file):
    dc.utils.download_url(SWEETLEAD_URL)
  loader = dc.data.CSVLoader(
      tasks=SWEET_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  # Initialize transformers
  transformers = [dc.trans.BalancingTransformer(dataset=dataset)]
  logger.info("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return SWEET_tasks, (dataset, None, None), transformers

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
      'task': dc.splits.TaskSplitter(),
      'stratified': dc.splits.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)

  if reload:
    dc.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                       transformers)
    all_dataset = (train, valid, test)

  return SWEET_tasks, (train, valid, test), transformers
