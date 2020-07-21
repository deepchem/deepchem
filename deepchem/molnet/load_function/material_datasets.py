"""
Datasets for inorganic crystal structures.
"""
import os
import logging
import deepchem
from deepchem.feat import Featurizer
from deepchem.trans import Transformer
from deepchem.splits.splitters import Splitter
from deepchem.molnet.defaults import get_defaults

from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# TODO: Change URLs
DEFAULT_DIR = deepchem.utils.get_data_dir()
BANDGAP_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/expt_gap.tar.gz'
PEROVSKITE_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/perovskite.tar.gz'

# dict of accepted featurizers for this dataset
# modify the returned dicts for your dataset
DEFAULT_FEATURIZERS = get_defaults("feat")

# Names of supported featurizers
featurizers = [
    'ElementPropertyFingerprint', 'SineCoulombMatrix',
    'StructureGraphFeaturizer'
]
DEFAULT_FEATURIZERS = {k: DEFAULT_FEATURIZERS[k] for k in featurizers}

# dict of accepted transformers
DEFAULT_TRANSFORMERS = get_defaults("trans")

# dict of accepted splitters
DEFAULT_SPLITTERS = get_defaults("splits")

# names of supported splitters
splitters = ['RandomSplitter']
DEFAULT_SPLITTERS = {k: DEFAULT_SPLITTERS[k] for k in splitters}


def load_bandgap(
    featurizer: Featurizer = DEFAULT_FEATURIZERS['ElementPropertyFingerprint'],
    transformers: List[Transformer] = [
        DEFAULT_TRANSFORMERS['NormalizationTransformer']
    ],
    splitter: Splitter = DEFAULT_SPLITTERS['RandomSplitter'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    featurizer_kwargs: Dict[str, object] = {'data_source': 'matminer'},
    splitter_kwargs: Dict[str, object] = {
        'frac_train': 0.8,
        'frac_valid': 0.1,
        'frac_test': 0.1
    },
    transformer_kwargs: Dict[str, Dict[str, object]] = {
        'NormalizationTransformer': {
            'transform_X': True
        }
    },
    **kwargs) -> Tuple[List, Tuple, List]:
  """Load band gap dataset.

  Contains 4604 experimentally measured band gaps for inorganic
  crystal structure compositions.
  
  Parameters
  ----------
  featurizer : ElementPropertyFingerprint
    A featurizer that inherits from deepchem.feat.Featurizer.
  transformers : List{List of allowed transformers for this dataset}
    A transformer that inherits from deepchem.trans.Transformer.
  splitter : RandomSplitter
    A splitter that inherits from deepchem.splits.splitters.Splitter.
  reload : bool (default True)
    Try to reload dataset from disk if already downloaded. Save to disk
    after featurizing.
  data_dir : str, optional
    Path to datasets.
  save_dir : str, optional
    Path to featurized datasets.
  featurizer_kwargs : dict
    Specify parameters to featurizer, e.g. {"size": 1024}
  splitter_kwargs : dict
    Specify parameters to splitter, e.g. {"seed": 42}
  transformer_kwargs : dict
    Maps transformer names to constructor arguments, e.g.
    {"BalancingTransformer": {"transform_x":True, "transform_y":False}}
  **kwargs : additional optional arguments.

  Returns
  -------
  tasks, datasets, transformers : tuple
    tasks : list
      Column names corresponding to machine learning target variables.
    datasets : tuple
      train, validation, test splits of data as
      ``deepchem.data.datasets.Dataset`` instances.
    transformers : list
      ``deepchem.trans.transformers.Transformer`` instances applied
      to dataset.

  References
  ----------
  .. [1] Zhuo, Y. et al. "Predicting the Band Gaps of Inorganic Solids by Machine Learning." J. Phys. Chem. Lett. (2018) DOI: 10.1021/acs.jpclett.8b00124.

  .. [2] Dunn, A. et al. "Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm." https://arxiv.org/abs/2005.00707 (2020)

  Examples
  --------
  >> import deepchem as dc
  >> tasks, datasets, transformers = dc.molnet.load_bandgap(reload=False)
  >> train_dataset, val_dataset, test_dataset = datasets
  >> n_tasks = len(tasks)
  >> n_features = train_dataset.get_data_shape()[0]
  >> model = dc.models.MultitaskRegressor(n_tasks, n_features)

  """

  # Featurize
  logger.info("About to featurize band gap dataset.")
  my_tasks = ['gap expt']  # machine learning targets

  # Get DeepChem data directory if needed
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  # Check for str args to featurizer and splitter
  if isinstance(featurizer, str):
    featurizer = DEFAULT_FEATURIZERS[featurizer](**featurizer_kwargs)
  elif issubclass(featurizer, Featurizer):
    featurizer = featurizer(**featurizer_kwargs)

  if isinstance(splitter, str):
    splitter = DEFAULT_SPLITTERS[splitter]()
  elif issubclass(splitter, Splitter):
    splitter = splitter()

  # Reload from disk
  if reload:
    featurizer_name = str(featurizer.__class__.__name__)
    splitter_name = str(splitter.__class__.__name__)
    save_folder = os.path.join(save_dir, "bandgap-featurized", featurizer_name,
                               splitter_name)

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return my_tasks, all_dataset, transformers

  # First type of supported featurizers
  supported_featurizers = ['ElementPropertyFingerprint'
                          ]  # type: List[Featurizer]

  # Load .tar.gz file
  if featurizer.__class__.__name__ in supported_featurizers:
    dataset_file = os.path.join(data_dir, 'expt_gap.tar.gz')
    deepchem.utils.untargz_file(dataset_file, dest_dir=data_dir)
    dataset_file = os.path.join(data_dir, 'expt_gap.json')

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=BANDGAP_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'expt_gap.tar.gz'), data_dir)

    # Changer loader to match featurizer and data file type
    loader = deepchem.data.JsonLoader(
        tasks=my_tasks,
        feature_field="composition",
        label_field="gap expt",
        featurizer=featurizer)

  # Featurize dataset
  dataset = loader.create_dataset(dataset_file)

  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset, **splitter_kwargs)

  # Initialize transformers
  transformers = [
      DEFAULT_TRANSFORMERS[t](dataset=dataset, **transformer_kwargs[t])
      if isinstance(t, str) else t(
          dataset=dataset, **transformer_kwargs[str(t.__name__)])
      for t in transformers
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  if reload:  # save to disk
    deepchem.utils.save.save_dataset_to_disk(
        save_folder, train_dataset, valid_dataset, test_dataset, transformers)

  return my_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_perovskite(
    featurizer: Featurizer = DEFAULT_FEATURIZERS['SineCoulombMatrix'],
    transformers: List[Transformer] = [
        DEFAULT_TRANSFORMERS['NormalizationTransformer']
    ],
    splitter: Splitter = DEFAULT_SPLITTERS['RandomSplitter'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    featurizer_kwargs: Dict[str, object] = None,
    splitter_kwargs: Dict[str, object] = {
        'frac_train': 0.6,
        'frac_valid': 0.2,
        'frac_test': 0.2
    },
    transformer_kwargs: Dict[str, Dict[str, object]] = {
        'NormalizationTransformer': {
            'transform_X': True
        }
    },
    **kwargs) -> Tuple[List, Tuple, List]:
  """Load perovskite dataset.

  Contains 18928 perovskite structures and their formation energies.
  
  Parameters
  ----------
  featurizer : StructureGraphFeaturizer
    A featurizer that inherits from deepchem.feat.Featurizer.
  transformers : List{List of allowed transformers for this dataset}
    A transformer that inherits from deepchem.trans.Transformer.
  splitter : RandomSplitter
    A splitter that inherits from deepchem.splits.splitters.Splitter.
  reload : bool (default True)
    Try to reload dataset from disk if already downloaded. Save to disk
    after featurizing.
  data_dir : str, optional
    Path to datasets.
  save_dir : str, optional
    Path to featurized datasets.
  featurizer_kwargs : dict
    Specify parameters to featurizer, e.g. {"size": 1024}
  splitter_kwargs : dict
    Specify parameters to splitter, e.g. {"seed": 42}
  transformer_kwargs : dict
    Maps transformer names to constructor arguments, e.g.
    {"BalancingTransformer": {"transform_x":True, "transform_y":False}}
  **kwargs : additional optional arguments.

  Returns
  -------
  tasks, datasets, transformers : tuple
    tasks : list
      Column names corresponding to machine learning target variables.
    datasets : tuple
      train, validation, test splits of data as
      ``deepchem.data.datasets.Dataset`` instances.
    transformers : list
      ``deepchem.trans.transformers.Transformer`` instances applied
      to dataset.

  References
  ----------
  .. [1] Castelli, I. et al. "New cubic perovskites for one- and two-photon water splitting using the computational materials repository." Energy Environ. Sci., (2012), 5, 9034-9043 DOI: 10.1039/C2EE22341D.

  .. [2] Dunn, A. et al. "Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm." https://arxiv.org/abs/2005.00707 (2020)

  Examples
  --------
  >> import deepchem as dc
  >> tasks, datasets, transformers = dc.molnet.load_perovskite(reload=False)
  >> train_dataset, val_dataset, test_dataset = datasets
  >> n_tasks = len(tasks)
  >> n_features = train_dataset.get_data_shape()[0]
  >> model = dc.models.MultitaskRegressor(n_tasks, n_features)

  """

  # Featurize
  logger.info("About to featurize perovskite dataset.")
  my_tasks = ['e_form']  # machine learning targets

  # Get DeepChem data directory if needed
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  # Check for str args to featurizer and splitter
  if isinstance(featurizer, str):
    featurizer = DEFAULT_FEATURIZERS[featurizer](**featurizer_kwargs)
  elif issubclass(featurizer, Featurizer):
    featurizer = featurizer(**featurizer_kwargs)

  if isinstance(splitter, str):
    splitter = DEFAULT_SPLITTERS[splitter]()
  elif issubclass(splitter, Splitter):
    splitter = splitter()

  # Reload from disk
  if reload:
    featurizer_name = str(featurizer.__class__.__name__)
    splitter_name = str(splitter.__class__.__name__)
    save_folder = os.path.join(save_dir, "perovskite-featurized",
                               featurizer_name, splitter_name)

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return my_tasks, all_dataset, transformers

  # First type of supported featurizers
  supported_featurizers = ['StructureGraphFeaturizer',
                           'SineCoulombMatrix']  # type: List[Featurizer]

  # Load .tar.gz file
  if featurizer.__class__.__name__ in supported_featurizers:
    dataset_file = os.path.join(data_dir, 'perovskite.tar.gz')
    deepchem.utils.untargz_file(dataset_file, dest_dir=data_dir)
    dataset_file = os.path.join(data_dir, 'perovskite.json')

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=PEROVSKITE_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'perovskite.tar.gz'), data_dir)

    # Changer loader to match featurizer and data file type
    loader = deepchem.data.JsonLoader(
        tasks=my_tasks,
        feature_field="structure",
        label_field="e_form",
        featurizer=featurizer)

  # Featurize dataset
  dataset = loader.create_dataset(dataset_file)

  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset, **splitter_kwargs)

  # Initialize transformers
  transformers = [
      DEFAULT_TRANSFORMERS[t](dataset=dataset, **transformer_kwargs[t])
      if isinstance(t, str) else t(
          dataset=dataset, **transformer_kwargs[str(t.__name__)])
      for t in transformers
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  if reload:  # save to disk
    deepchem.utils.save.save_dataset_to_disk(
        save_folder, train_dataset, valid_dataset, test_dataset, transformers)

  return my_tasks, (train_dataset, valid_dataset, test_dataset), transformers
