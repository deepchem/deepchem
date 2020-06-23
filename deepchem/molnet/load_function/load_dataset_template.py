"""
Short docstring description of dataset.
"""
import os
import logging
import deepchem

from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
MYDATASET_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/mydataset.tar.gz'
MYDATASET_CSV_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/mydataset.csv'


def load_mydataset(featurizer: str = None,
                   split: str = 'random',
                   reload: bool = True,
                   move_mean: bool = True,
                   data_dir: str = None,
                   save_dir: str = None,
                   **kwargs) -> Iterable:
  """Load mydataset.

  This is a template for adding a function to load a dataset from
  MoleculeNet. Adjust the global variable URL strings, default parameters,
  and variable names as needed. The function will need to be modified
  to handle the allowed featurizers for your dataset. 

  If `reload = True` and `data_dir` (`save_dir`) is specified, the loader
  will attempt to load the raw dataset (featurized dataset) from disk.
  Otherwise, the dataset will be downloaded from the DeepChem AWS bucket.

  The dataset will be featurized with `featurizer` and separated into
  train/val/test sets according to `split`. Additional kwargs may
  be given for specific featurizers and splitters.

  Please refer to the MoleculeNet documentation for further information
  https://deepchem.readthedocs.io/en/latest/moleculenet.html.
  
  Parameters
  ----------
  featurizer: {List of allowed featurizers for this dataset}
    A featurizer that inherits from deepchem.feat.Featurizer.
  split: {'random', 'stratified', 'index', 'scaffold'}
    A splitter that inherits from deepchem.splits.splitters.Splitter.
  reload: bool (default True)
    Try to reload dataset from disk if already downloaded. Save to disk
    after featurizing.
  move_mean: bool (default True)
    Center data to have 0 mean after transform.
  data_dir: str, optional
    Path to datasets.
  save_dir: str, optional
    Path to featurized datasets.
  **kwargs: optional arguments to featurizers and splitters.

  References
  ----------
  MLA style references for this dataset. E.g.
    Wu, Zhenqin et al. "MoleculeNet: a benchmark for molecular
      machine learning." Chemical Science, vol. 9, 2018, 
      pp. 513-530, 10.1039/c7sc02664a.

    Last, First et al. "Article title." Journal name, vol. #,
      no. #, year, pp. page range, DOI. 

  """

  # Featurize mydataset
  logger.info("About to featurize mydataset.")
  my_tasks = ["task1", "task2", "task3"]  # machine learning targets

  # Get DeepChem data directory if needed
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  # Reload from disk
  if reload:
    save_folder = os.path.join(save_dir, "mydataset-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder,
                                 str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return my_tasks, all_dataset, transformers

  sdf_featurizers = []  # e.g. 'CoulombMatrix' or 'MP'

  # If featurizer requires a non-CSV file format, load .tar.gz file
  if featurizer in sdf_featurizers:
    dataset_file = os.path.join(data_dir, 'mydataset.sdf')

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MYDATASET_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'mydataset.tar.gz'), data_dir)
  else:  # only load CSV file
    dataset_file = os.path.join(data_dir, "mydataset.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(
          url=MYDATASET_CSV_URL, dest_dir=data_dir)

  # Handle all allowed SDF featurizers
  if featurizer in sdf_featurizers:
    if featurizer == 'Featurizer1':
      featurizer = deepchem.feat.Featurizer1()
    elif featurizer == 'Featurizer2':
      featurizer = deepchem.feat.Featurizer2()

    loader = deepchem.data.SDFLoader(
      tasks=my_tasks,
      smiles_field="smiles",  # column name holding SMILES strings
      mol_field="mol",  # field where RKit mol objects are stored
      featurizer=featurizer)
  else:  # Handle allowed CSV featurizers
    if featurizer == 'Featurizer3':
      featurizer = deepchem.feat.Featurizer3()
    elif featurizer == 'Featurizer4':
      featurizer = deepchem.feat.Featurizer4()

    loader = deepchem.data.CSVLoader(
        tasks=my_tasks, smiles_field="smiles", featurizer=featurizer)

  # Featurize dataset
  dataset = loader.featurize(dataset_file)
  if split is None:  # Must give a recommended split for data
    raise ValueError()

  # Generate Splitter
  splitters = {
      'index':
      deepchem.splits.IndexSplitter(),
      'random':
      deepchem.splits.RandomSplitter(),
      'stratified':
      deepchem.splits.SingletaskStratifiedSplitter(
          task_number=len(my_tasks)),
      'scaffold':
      deepchem.splits.ScaffoldSplitter()
  }

  splitter = splitters[split]

  # 80/10/10 train/val/test split is default
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset, move_mean=move_mean)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  if reload:  # save to disk
    deepchem.utils.save.save_dataset_to_disk(save_folder, train_dataset,
                                             valid_dataset, test_dataset,
                                             transformers)

  return my_tasks, (train_dataset, valid_dataset, test_dataset), transformers
