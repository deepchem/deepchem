"""
Calculated formation energies for inorganic crystals from Materials Project.
"""
import os
import logging
import deepchem
from deepchem.feat import Featurizer, MaterialStructureFeaturizer, MaterialCompositionFeaturizer
from deepchem.trans import Transformer
from deepchem.splits.splitters import Splitter
from deepchem.molnet.defaults import get_defaults

from typing import List, Tuple, Dict, Optional, Union, Any, Type

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
MPFORME_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/mp_formation_energy.tar.gz'

# dict of accepted featurizers for this dataset
# modify the returned dicts for your dataset
DEFAULT_FEATURIZERS = get_defaults("feat")

# Names of supported featurizers
featurizers = [
    'CGCNNFeaturizer',
    'SineCoulombMatrix',
]
DEFAULT_FEATURIZERS = {k: DEFAULT_FEATURIZERS[k] for k in featurizers}

# dict of accepted transformers
DEFAULT_TRANSFORMERS = get_defaults("trans")

# dict of accepted splitters
DEFAULT_SPLITTERS = get_defaults("splits")

# names of supported splitters
splitters = ['RandomSplitter']
DEFAULT_SPLITTERS = {k: DEFAULT_SPLITTERS[k] for k in splitters}


def load_mp_formation_energy(
    featurizer=DEFAULT_FEATURIZERS['SineCoulombMatrix'],
    transformers: List = [DEFAULT_TRANSFORMERS['NormalizationTransformer']],
    splitter=DEFAULT_SPLITTERS['RandomSplitter'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    featurizer_kwargs: Dict[str, Any] = {},
    splitter_kwargs: Dict[str, Any] = {
        'frac_train': 0.8,
        'frac_valid': 0.1,
        'frac_test': 0.1
    },
    transformer_kwargs: Dict[str, Dict[str, Any]] = {
        'NormalizationTransformer': {
            'transform_X': True
        }
    },
    **kwargs) -> Tuple[List, Tuple, List]:
  """Load mp formation energy dataset.

  Contains 132752 calculated formation energies and inorganic
  crystal structures from the Materials Project database. In benchmark
  studies, random forest models achieved a mean average error of
  0.116 eV/atom during five-folded nested cross validation on this
  dataset.

  For more details on the dataset see [1]_. For more details
  on previous benchmarks for this dataset, see [2]_.
  
  Parameters
  ----------
  featurizer : MaterialCompositionFeaturizer 
    (default CGCNNFeaturizer)
    A featurizer that inherits from deepchem.feat.Featurizer.
  transformers : List[Transformer]
    A transformer that inherits from deepchem.trans.Transformer.
  splitter : Splitter (default RandomSplitter)
    A splitter that inherits from deepchem.splits.splitters.Splitter.
  reload : bool (default True)
    Try to reload dataset from disk if already downloaded. Save to disk
    after featurizing.
  data_dir : str, optional
    Path to datasets.
  save_dir : str, optional
    Path to featurized datasets.
  featurizer_kwargs : Dict[str, Any]
    Specify parameters to featurizer, e.g. {"size": 1024}
  splitter_kwargs : Dict[str, Any]
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
  .. [1] A. Jain*, S.P. Ong*, et al. (*=equal contributions) The Materials Project: A materials genome approach to accelerating materials innovation APL Materials, 2013, 1(1), 011002. doi:10.1063/1.4812323 (2013).

  .. [2] Dunn, A. et al. "Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm." https://arxiv.org/abs/2005.00707 (2020)

  Examples
  --------
  >> import deepchem as dc
  >> tasks, datasets, transformers = dc.molnet.load_mp_formation_energy(reload=False)
  >> train_dataset, val_dataset, test_dataset = datasets
  >> n_tasks = len(tasks)
  >> n_features = train_dataset.get_data_shape()[0]
  >> model = dc.models.MultitaskRegressor(n_tasks, n_features)

  """

  # Featurize
  logger.info("About to featurize mp formation energy dataset.")
  my_tasks = ['formation_energy']  # machine learning targets

  # Get DeepChem data directory if needed
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if issubclass(featurizer, MaterialStructureFeaturizer):
    featurizer = featurizer(**featurizer_kwargs)
  else:
    raise TypeError(
        "featurizer must be a subclass of MaterialStructureFeaturizer.")

  if issubclass(splitter, Splitter):
    splitter = splitter()
  else:
    raise TypeError("splitter must be a subclass of Splitter.")

  # Reload from disk
  if reload:
    featurizer_name = str(featurizer.__class__.__name__)
    splitter_name = str(splitter.__class__.__name__)
    save_folder = os.path.join(save_dir, "mp-forme-featurized", featurizer_name,
                               splitter_name)

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return my_tasks, all_dataset, transformers

  # First type of supported featurizers
  supported_featurizers: List[str] = [
      'CGCNNFeaturizer',
      'SineCoulombMatrix',
  ]

  # Load .tar.gz file
  if featurizer.__class__.__name__ in supported_featurizers:
    dataset_file = os.path.join(data_dir, 'mp_formation_energy.json')

    if not os.path.exists(dataset_file):
      targz_file = os.path.join(data_dir, 'mp_formation_energy.tar.gz')
      if not os.path.exists(targz_file):
        deepchem.utils.download_url(url=MPFORME_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'mp_formation_energy.tar.gz'), data_dir)

    # Changer loader to match featurizer and data file type
    loader = deepchem.data.JsonLoader(
        tasks=my_tasks,
        feature_field="structure",
        label_field="formation_energy",
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
