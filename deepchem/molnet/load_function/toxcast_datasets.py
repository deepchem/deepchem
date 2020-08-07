"""
TOXCAST dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
TOXCAST_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"


def load_toxcast(featurizer='ECFP',
                 split='index',
                 reload=True,
                 data_dir=None,
                 save_dir=None,
                 **kwargs):
  """Load Toxcast dataset

  ToxCast is an extended data collection from the same
  initiative as Tox21, providing toxicology data for a large
  library of compounds based on in vitro high-throughput
  screening. The processed collection includes qualitative
  results of over 600 experiments on 8k compounds.

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "smiles": SMILES representation of the molecular structure
  - "ACEA_T47D_80hr_Negative" ~ "Tanguay_ZF_120hpf_YSE_up": Bioassays results.
    Please refer to the section "high-throughput assay information" at 
    https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data 
    for details.

  References
  ----------
  .. [1] Richard, Ann M., et al. "ToxCast chemical landscape: paving the road 
     to 21st century toxicology." Chemical research in toxicology 29.8 (2016):
     1225-1251.
  """
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "toxcast-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  dataset_file = os.path.join(data_dir, "toxcast_data.csv.gz")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=TOXCAST_URL, dest_dir=data_dir)

  dataset = deepchem.utils.save.load_from_disk(dataset_file)
  logger.info("Columns of dataset: %s" % str(dataset.columns.values))
  logger.info("Number of examples in dataset: %s" % str(dataset.shape[0]))
  TOXCAST_tasks = dataset.columns.values[1:].tolist()

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return TOXCAST_tasks, all_dataset, transformers

  # Featurize TOXCAST dataset
  logger.info("About to featurize TOXCAST dataset.")

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
      tasks=TOXCAST_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  if split == None:
    transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]
    logger.info("Split is None, about to transform data.")
    for transformer in transformers:
      dataset = transformer.transform(dataset)
    return TOXCAST_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
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

  transformers = [deepchem.trans.BalancingTransformer(dataset=train)]

  logger.info("About to transform dataset.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)

  return TOXCAST_tasks, (train, valid, test), transformers
