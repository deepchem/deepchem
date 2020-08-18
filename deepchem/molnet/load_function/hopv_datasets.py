"""
HOPV dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

HOPV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hopv.tar.gz"
DEFAULT_DIR = deepchem.utils.get_data_dir()


def load_hopv(featurizer='ECFP',
              split='index',
              reload=True,
              data_dir=None,
              save_dir=None,
              **kwargs):
  """Load HOPV datasets. Does not do train/test split

  The HOPV datasets consist of the "Harvard Organic
  Photovoltaic Dataset. This dataset includes 350 small
  molecules and polymers that were utilized as p-type materials
  in OPVs. Experimental properties include: HOMO [a.u.], LUMO
  [a.u.], Electrochemical gap [a.u.], Optical gap [a.u.], Power
  conversion efficiency [%], Open circuit potential [V], Short
  circuit current density [mA/cm^2], and fill factor [%].
  Theoretical calculations in the original dataset have been
  removed (for now).

  Lopez, Steven A., et al. "The Harvard organic photovoltaic dataset." Scientific data 3.1 (2016): 1-7.
  """
  # Featurize HOPV dataset
  logger.info("About to featurize HOPV dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  hopv_tasks = [
      'HOMO', 'LUMO', 'electrochemical_gap', 'optical_gap', 'PCE', 'V_OC',
      'J_SC', 'fill_factor'
  ]

  if reload:
    save_folder = os.path.join(save_dir, "hopv-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return hopv_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "hopv.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=HOPV_URL, dest_dir=data_dir)
    deepchem.utils.untargz_file(os.path.join(data_dir, 'hopv.tar.gz'), data_dir)

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
      tasks=hopv_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split == None:
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return hopv_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter()
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

  transformers = [
      deepchem.trans.NormalizationTransformer(transform_y=True, dataset=train)
  ]

  logger.info("About to transform data.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return hopv_tasks, (train, valid, test), transformers
