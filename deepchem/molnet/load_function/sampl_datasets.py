"""
SAMPL dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

SAMPL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
DEFAULT_DIR = deepchem.utils.get_data_dir()


def load_sampl(featurizer='ECFP',
               split='index',
               reload=True,
               move_mean=True,
               data_dir=None,
               save_dir=None,
               **kwargs):
  """Load SAMPL(FreeSolv) dataset

  The Free Solvation Database, FreeSolv(SAMPL), provides experimental and 
  calculated hydration free energy of small molecules in water. The calculated
  values are derived from alchemical free energy calculations using molecular
  dynamics simulations. The experimental values are included in the benchmark
  collection.

  Random splitting is recommended for this dataset.
  
  The raw data csv file contains columns below:

  - "iupac" - IUPAC name of the compound
  - "smiles" - SMILES representation of the molecular structure
  - "expt" - Measured solvation energy (unit: kcal/mol) of the compound, 
    used as label
  - "calc" - Calculated solvation energy (unit: kcal/mol) of the compound


  References
  ----------
  .. [1] Mobley, David L., and J. Peter Guthrie. "FreeSolv: a database of 
     experimental and calculated hydration free energies, with input files."
     Journal of computer-aided molecular design 28.7 (2014): 711-720.
  """
  # Featurize SAMPL dataset
  logger.info("About to featurize SAMPL dataset.")
  logger.info("About to load SAMPL dataset.")

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "sampl-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  dataset_file = os.path.join(data_dir, "SAMPL.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=SAMPL_URL, dest_dir=data_dir)

  SAMPL_tasks = ['expt']

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return SAMPL_tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == 'smiles2img':
    img_size = kwargs.get("img_size", 80)
    img_spec = kwargs.get("img_spec", "std")
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  loader = deepchem.data.CSVLoader(
      tasks=SAMPL_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split == None:
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=move_mean)
    ]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return SAMPL_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
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

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train, move_mean=move_mean)
  ]

  logger.info("About to transform dataset.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return SAMPL_tasks, (train, valid, test), transformers
