"""
Lipophilicity dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
LIPO_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"


def load_lipo(featurizer='ECFP',
              split='index',
              reload=True,
              move_mean=True,
              data_dir=None,
              save_dir=None,
              **kwargs):
  """Load Lipophilicity dataset

  Lipophilicity is an important feature of drug molecules that affects both 
  membrane permeability and solubility. The lipophilicity dataset, curated 
  from ChEMBL database, provides experimental results of octanol/water 
  distribution coefficient (logD at pH 7.4) of 4200 compounds.

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "smiles" - SMILES representation of the molecular structure
  - "exp" - Measured octanol/water distribution coefficient (logD) of the 
    compound, used as label

  References
  ----------
  .. [1] Hersey, A. ChEMBL Deposited Data Set - AZ dataset; 2015. 
     https://doi.org/10.6019/chembl3301361
  """
  # Featurize Lipophilicity dataset
  logger.info("About to featurize Lipophilicity dataset.")
  logger.info("About to load Lipophilicity dataset.")

  Lipo_tasks = ['exp']

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "lipo-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return Lipo_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "Lipophilicity.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=LIPO_URL, dest_dir=data_dir)

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
      tasks=Lipo_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split is None:
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=move_mean)
    ]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return Lipo_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split data with {} splitter.".format(split))
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train, move_mean=move_mean)
  ]

  logger.info("About to transform data.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return Lipo_tasks, (train, valid, test), transformers
