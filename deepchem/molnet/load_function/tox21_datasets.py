"""
Tox21 dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
DEFAULT_DIR = deepchem.utils.get_data_dir()


def load_tox21(featurizer='ECFP',
               split='index',
               reload=True,
               K=4,
               data_dir=None,
               save_dir=None,
               **kwargs):
  """Load Tox21 dataset

  The "Toxicology in the 21st Century" (Tox21) initiative created a public
  database measuring toxicity of compounds, which has been used in the 2014
  Tox21 Data Challenge. This dataset contains qualitative toxicity measurements
  for 8k compounds on 12 different targets, including nuclear receptors and
  stress response pathways.

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:

  - "smiles" - SMILES representation of the molecular structure
  - "NR-XXX" - Nuclear receptor signaling bioassays results
  - "SR-XXX" - Stress response bioassays results

  please refer to https://tripod.nih.gov/tox21/challenge/data.jsp for details.

  References
  ----------
  .. [1] Tox21 Challenge. https://tripod.nih.gov/tox21/challenge/
  """
  # Featurize Tox21 dataset

  tox21_tasks = [
      'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
      'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
  ]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "tox21-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return tox21_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "tox21.csv.gz")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=TOX21_URL, dest_dir=data_dir)

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == 'AdjacencyConv':
    featurizer = deepchem.feat.AdjacencyFingerprint(
        max_n_atoms=150, max_valence=6)
  elif featurizer == "smiles2img":
    img_size = kwargs.get("img_size", 80)
    img_spec = kwargs.get("img_spec", "std")
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  loader = deepchem.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split == None:
    # Initialize transformers
    transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]

    logger.info("About to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return tox21_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter(),
      'task': deepchem.splits.TaskSplitter(),
      'stratified': deepchem.splits.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  if split == 'task':
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
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

    transformers = [deepchem.trans.BalancingTransformer(dataset=train)]

    logger.info("About to transform data")
    for transformer in transformers:
      train = transformer.transform(train)
      valid = transformer.transform(valid)
      test = transformer.transform(test)

    if reload:
      deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                               transformers)
  return tox21_tasks, all_dataset, transformers
