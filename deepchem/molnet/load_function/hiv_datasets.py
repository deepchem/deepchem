"""
hiv dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

HIV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
DEFAULT_DIR = deepchem.utils.get_data_dir()


def load_hiv(featurizer='ECFP',
             split='index',
             reload=True,
             data_dir=None,
             save_dir=None,
             **kwargs):
  """Load HIV dataset

  The HIV dataset was introduced by the Drug Therapeutics
  Program (DTP) AIDS Antiviral Screen, which tested the ability
  to inhibit HIV replication for over 40,000 compounds.
  Screening results were evaluated and placed into three
  categories: confirmed inactive (CI),confirmed active (CA) and
  confirmed moderately active (CM). We further combine the
  latter two labels, making it a classification task between
  inactive (CI) and active (CA and CM).

  Scaffold splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "smiles": SMILES representation of the molecular structure
  - "activity": Three-class labels for screening results: CI/CM/CA
  - "HIV_active": Binary labels for screening results: 1 (CA/CM) and 0 (CI)

  References
  ----------
  .. [1] AIDS Antiviral Screen Data. 
     https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data
  """
  # Featurize hiv dataset
  logger.info("About to featurize hiv dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  hiv_tasks = ["HIV_active"]

  if reload:
    save_folder = os.path.join(save_dir, "hiv-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return hiv_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "HIV.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=HIV_URL, dest_dir=data_dir)

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == "smiles2img":
    img_spec = kwargs.get("img_spec", "std")
    img_size = kwargs.get("img_size", 80)
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  loader = deepchem.data.CSVLoader(
      tasks=hiv_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split is None:
    transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return hiv_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter(),
      'stratified': deepchem.splits.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split dataset with {} splitter.".format(split))
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)
  train, valid, test = splitter.train_valid_test_split(dataset)

  transformers = [deepchem.trans.BalancingTransformer(dataset=train)]

  logger.info("About to transform data.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return hiv_tasks, (train, valid, test), transformers
