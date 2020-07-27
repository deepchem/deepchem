"""
MUV dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
MUV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"


def load_muv(featurizer='ECFP',
             split='index',
             reload=True,
             K=4,
             data_dir=None,
             save_dir=None,
             **kwargs):
  """Load MUV dataset

  The Maximum Unbiased Validation (MUV) group is a benchmark dataset selected
  from PubChem BioAssay by applying a refined nearest neighbor analysis.

  The MUV dataset contains 17 challenging tasks for around 90 thousand
  compounds and is specifically designed for validation of virtual screening
  techniques.

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "mol_id" - PubChem CID of the compound
  - "smiles" - SMILES representation of the molecular structure
  - "MUV-XXX" - Measured results (Active/Inactive) for bioassays

  References
  ----------
  .. [1] Rohrer, Sebastian G., and Knut Baumann. "Maximum unbiased validation 
     (MUV) data sets for virtual screening based on PubChem bioactivity data." 
     Journal of chemical information and modeling 49.2 (2009): 169-184.
  """
  # Load MUV dataset
  logger.info("About to load MUV dataset.")

  MUV_tasks = sorted([
      'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548',
      'MUV-852', 'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858',
      'MUV-713', 'MUV-733', 'MUV-652', 'MUV-466', 'MUV-832'
  ])

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "muv-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return MUV_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "muv.csv.gz")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=MUV_URL, dest_dir=data_dir)

  # Featurize MUV dataset
  logger.info("About to featurize MUV dataset.")

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
      tasks=MUV_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  if split == None:
    transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return MUV_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'task': deepchem.splits.TaskSplitter(),
      'stratified': deepchem.splits.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  if split == 'task':
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
    logger.info(
        "K-Fold split complete. Use the transformers for this dataset on the returned folds."
    )
    return MUV_tasks, all_dataset, []

  else:
    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train, valid, test = splitter.train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test)
    all_dataset = (train, valid, test)
    if reload:
      deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                               transformers)
    return MUV_tasks, all_dataset, transformers
