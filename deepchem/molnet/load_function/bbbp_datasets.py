"""
Blood-Brain Barrier Penetration dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"


def load_bbbp(featurizer='ECFP',
              split='random',
              reload=True,
              data_dir=None,
              save_dir=None,
              **kwargs):
  """Load BBBP dataset

  The blood-brain barrier penetration (BBBP) dataset is designed for the
  modeling and prediction of barrier permeability. As a membrane separating
  circulating blood and brain extracellular fluid, the blood-brain barrier
  blocks most drugs, hormones and neurotransmitters. Thus penetration of the
  barrier forms a long-standing issue in development of drugs targeting
  central nervous system.

  This dataset includes binary labels for over 2000 compounds on their
  permeability properties.

  Scaffold splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "name" - Name of the compound
  - "smiles" - SMILES representation of the molecular structure
  - "p_np" - Binary labels for penetration/non-penetration

  References
  ----------
  .. [1] Martins, Ines Filipa, et al. "A Bayesian approach to in silico
     blood-brain barrier penetration modeling." Journal of chemical
     information and modeling 52.6 (2012): 1686-1697.
  """

  # Featurize bbb dataset
  logger.info("About to featurize bbbp dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  bbbp_tasks = ["p_np"]

  if reload:
    save_folder = os.path.join(save_dir, "bbbp-featurized", featurizer)
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return bbbp_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "BBBP.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=BBBP_URL, dest_dir=data_dir)

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
      tasks=bbbp_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split is None:
    # Initialize transformers
    transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return bbbp_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
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

  # Initialize transformers
  transformers = [deepchem.trans.BalancingTransformer(dataset=train)]

  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return bbbp_tasks, (train, valid, test), transformers
