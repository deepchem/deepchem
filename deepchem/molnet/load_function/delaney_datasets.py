"""
Delaney dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
DELANEY_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"


def load_delaney(featurizer='ECFP',
                 split='index',
                 reload=True,
                 move_mean=True,
                 data_dir=None,
                 save_dir=None,
                 **kwargs):
  """Load delaney dataset

  The Delaney(ESOL) dataset a regression dataset containing structures and 
  water solubility data for 1128 compounds. The dataset is widely used to 
  validate machine learning models on estimating solubility directly from 
  molecular structures (as encoded in SMILES strings).

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "Compound ID" - Name of the compound
  - "smiles" - SMILES representation of the molecular structure
  - "measured log solubility in mols per litre" - Log-scale water solubility 
    of the compound, used as label

  References
  ----------
  .. [1] Delaney, John S. "ESOL: estimating aqueous solubility directly from 
     molecular structure." Journal of chemical information and computer 
     sciences 44.3 (2004): 1000-1005.
  """
  # Featurize Delaney dataset
  logger.info("About to featurize Delaney dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR
  if reload:
    save_folder = os.path.join(save_dir, "delaney-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  dataset_file = os.path.join(data_dir, "delaney-processed.csv")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=DELANEY_URL, dest_dir=data_dir)

  delaney_tasks = ['measured log solubility in mols per litre']
  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return delaney_tasks, all_dataset, transformers

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
    res = kwargs.get("res", 0.5)
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec, res=res)

  loader = deepchem.data.CSVLoader(
      tasks=delaney_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split is None:
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=move_mean)
    ]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return delaney_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split dataset with {} splitter.".format(split))
  train, valid, test = splitter.train_valid_test_split(dataset)

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
  return delaney_tasks, (train, valid, test), transformers
