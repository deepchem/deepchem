"""
ZINC15 commercially-available compounds for virtual screening.
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

DEFAULT_DIR = deepchem.utils.data_utils.get_data_dir()
zinc15_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15.tar.gz"

# dict of accepted featurizers for this dataset
DEFAULT_FEATURIZERS = get_defaults("feat")

# Names of supported featurizers
zinc15_featurizers = [
    'SmilesToImage', 'OneHotFeaturizer', 'SmilesToSeq', 'RDKitDescriptors',
    'ConvMolFeaturizer', 'WeaveFeaturizer', 'CircularFingerprint',
    'Mol2VecFingerprint'
]
DEFAULT_FEATURIZERS = {k: DEFAULT_FEATURIZERS[k] for k in zinc15_featurizers}

# dict of accepted transformers
DEFAULT_TRANSFORMERS = get_defaults("trans")

# dict of accepted splitters
DEFAULT_SPLITTERS = get_defaults("splits")

# names of supported splitters
zinc15_splitters = ['RandomSplitter', 'RandomStratifiedSplitter']
DEFAULT_SPLITTERS = {k: DEFAULT_SPLITTERS[k] for k in zinc15_splitters}


def load_zinc15(
    featurizer=DEFAULT_FEATURIZERS['OneHotFeaturizer'],
    transformers: List = [DEFAULT_TRANSFORMERS['NormalizationTransformer']],
    splitter=DEFAULT_SPLITTERS['RandomSplitter'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    featurizer_kwargs: Dict[str, object] = {},
    splitter_kwargs: Dict[str, object] = {},
    transformer_kwargs: Dict[str, Dict[str, object]] = {
        'NormalizationTransformer': {
            'transform_X': True
        }
    },
    **kwargs) -> Tuple[List, Optional[Tuple], List]:
  """Load zinc15.

  ZINC15 is a dataset of over 230 million purchasable compounds for
  virtual screening of small molecules to identify structures that
  are likely to bind to drug targets. It is available with both 2D
  (SMILES string) and 3D representations, although only the 2D data
  is currently available through MolNet.

  If `reload = True` and `data_dir` (`save_dir`) is specified, the loader
  will attempt to load the raw dataset (featurized dataset) from disk.
  Otherwise, the dataset will be downloaded from the DeepChem AWS bucket.

  For more information on ZINC15, please see
  https://zinc15.docking.org/.

  Parameters
  ----------
  featurizer : allowed featurizers for this dataset
    A featurizer that inherits from deepchem.feat.Featurizer.
  transformers : List of allowed transformers for this dataset
    A transformer that inherits from deepchem.trans.Transformer.
  splitter : allowed splitters for this dataset
    A splitter that inherits from deepchem.splits.splitters.Splitter.
  reload : bool (default True)
    Try to reload dataset from disk if already downloaded. Save to disk
    after featurizing.
  data_dir : str, optional (default None)
    Path to datasets.
  save_dir : str, optional (default None)
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
  ...[1] Sterling and Irwin. J. Chem. Inf. Model, 2015 http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559.

  Examples
  --------
  >> import deepchem as dc
  >> tasks, datasets, transformers = dc.molnet.load_zinc15(reload=False)
  >> train_dataset, val_dataset, test_dataset = datasets
  >> n_tasks = len(tasks)
  >> n_features = train_dataset.get_data_shape()[0]
  >> model = dc.models.MultitaskRegressor(n_tasks, n_features)

  """

  # Featurize zinc15
  logger.info("About to featurize zinc15.")
  my_tasks = ['mwt', 'logp', 'reactive']  # machine learning targets

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
    save_folder = os.path.join(save_dir, "zinc15-featurized", featurizer_name,
                               splitter_name)

    loaded, all_dataset, transformers = deepchem.utils.data_utils.load_dataset_from_disk(
        save_folder)
    if loaded:
      return my_tasks, all_dataset, transformers

  if str(featurizer.__class__.__name__) in zinc15_featurizers:
    dataset_file = os.path.join(data_dir, 'zinc15.tar.gz')

    if not os.path.exists(dataset_file):
      deepchem.utils.data_utils.download_url(url=zinc15_URL, dest_dir=data_dir)

    deepchem.utils.data_utils.untargz_file(
        os.path.join(data_dir, 'zinc15.tar.gz'), data_dir)
    dataset_file = 'zinc15.csv'

    loader = deepchem.data.CSVLoader(
        tasks=my_tasks,
        feature_field="smiles",
        id_field='zinc_id',
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
    deepchem.utils.data_utils.save_dataset_to_disk(
        save_folder, train_dataset, valid_dataset, test_dataset, transformers)

  return my_tasks, (train_dataset, valid_dataset, test_dataset), transformers
