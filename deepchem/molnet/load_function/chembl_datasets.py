"""
ChEMBL dataset loader.
"""
import os
import logging
import deepchem
from deepchem.molnet.load_function.chembl_tasks import chembl_tasks

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()


def load_chembl(shard_size=2000,
                featurizer="ECFP",
                set="5thresh",
                split="random",
                reload=True,
                data_dir=None,
                save_dir=None,
                **kwargs):

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  logger.info("About to load ChEMBL dataset.")

  if reload:
    save_folder = os.path.join(save_dir, "chembl-featurized", featurizer)
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return chembl_tasks, all_dataset, transformers

  dataset_path = os.path.join(data_dir, "chembl_%s.csv.gz" % set)
  if not os.path.exists(dataset_path):
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_5thresh.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_sparse.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_test.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_train.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_5thresh_ts_valid.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_test.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_train.csv.gz",
        dest_dir=data_dir)
    deepchem.utils.download_url(
        url=
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_year_sets/chembl_sparse_ts_valid.csv.gz",
        dest_dir=data_dir)

  if split == "year":
    train_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_train.csv.gz" % set)
    valid_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_valid.csv.gz" % set)
    test_files = os.path.join(
        data_dir, "./chembl_year_sets/chembl_%s_ts_test.csv.gz" % set)

  # Featurize ChEMBL dataset
  logger.info("About to featurize ChEMBL dataset.")
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
      tasks=chembl_tasks, smiles_field="smiles", featurizer=featurizer)

  if split == "year":
    logger.info("Featurizing train datasets")
    train = loader.featurize(train_files, shard_size=shard_size)
    logger.info("Featurizing valid datasets")
    valid = loader.featurize(valid_files, shard_size=shard_size)
    logger.info("Featurizing test datasets")
    test = loader.featurize(test_files, shard_size=shard_size)
  else:
    dataset = loader.featurize(dataset_path, shard_size=shard_size)

  if split is None:
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]

    logger.info("Split is None, about to transform data.")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return chembl_tasks, (dataset, None, None), transformers

  if split != "year":
    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'scaffold': deepchem.splits.ScaffoldSplitter(),
    }

    splitter = splitters[split]
    logger.info("Performing new split.")
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
  return chembl_tasks, (train, valid, test), transformers
