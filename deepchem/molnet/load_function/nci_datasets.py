"""
NCI dataset loader.
Original Author - Bharath Ramsundar
Author - Aneesh Pappu
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
NCI_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/nci_unique.csv"


def load_nci(featurizer='ECFP',
             shard_size=1000,
             split='random',
             reload=True,
             data_dir=None,
             save_dir=None,
             **kwargs):

  # Load nci dataset
  logger.info("About to load NCI dataset.")

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  all_nci_tasks = [
      'CCRF-CEM', 'HL-60(TB)', 'K-562', 'MOLT-4', 'RPMI-8226', 'SR',
      'A549/ATCC', 'EKVX', 'HOP-62', 'HOP-92', 'NCI-H226', 'NCI-H23',
      'NCI-H322M', 'NCI-H460', 'NCI-H522', 'COLO 205', 'HCC-2998', 'HCT-116',
      'HCT-15', 'HT29', 'KM12', 'SW-620', 'SF-268', 'SF-295', 'SF-539',
      'SNB-19', 'SNB-75', 'U251', 'LOX IMVI', 'MALME-3M', 'M14', 'MDA-MB-435',
      'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'UACC-257', 'UACC-62', 'IGR-OV1',
      'OVCAR-3', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'NCI/ADR-RES', 'SK-OV-3',
      '786-0', 'A498', 'ACHN', 'CAKI-1', 'RXF 393', 'SN12C', 'TK-10', 'UO-31',
      'PC-3', 'DU-145', 'MCF7', 'MDA-MB-231/ATCC', 'MDA-MB-468', 'HS 578T',
      'BT-549', 'T-47D'
  ]

  if reload:
    save_folder = os.path.join(save_dir, "nci-featurized", featurizer)
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return all_nci_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "nci_unique.csv")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=NCI_URL, dest_dir=data_dir)

  # Featurize nci dataset
  logger.info("About to featurize nci dataset.")
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
      tasks=all_nci_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file, shard_size=shard_size)

  if split == None:
    logger.info("Split is None, about to transform data")
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]
    for transformer in transformers:
      dataset = transformer.transform(dataset)
    return all_nci_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split data with {} splitter.".format(splitter))
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

  logger.info("About to transform dataset.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return all_nci_tasks, (train, valid, test), transformers
