"""
ZINC15 commercially-available compounds for virtual screening.
"""
import os
import logging
import numpy as np
import deepchem
from deepchem.feat import Featurizer
from deepchem.trans import Transformer
from deepchem.splits.splitters import Splitter
from deepchem.molnet.defaults import get_defaults

from typing import List, Tuple, Dict, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.data_utils.get_data_dir()

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
    dataset_size: str = '250K',
    dataset_dimension: str = '2D',
    test_run: bool = False) -> Tuple[List, Optional[Tuple], List]:
  """Load zinc15.

  ZINC15 is a dataset of over 230 million purchasable compounds for
  virtual screening of small molecules to identify structures that
  are likely to bind to drug targets. ZINC15 data is currently available
  in 2D (SMILES string) format.

  MolNet provides subsets of 250K, 1M, and 10M "lead-like" compounds
  from ZINC15. The full dataset of 270M "goldilocks" compounds is also
  available. Compounds in ZINC15 are labeled by their molecular weight 
  and LogP (solubility) values. Each compound also has information about how
  readily available (purchasable) it is and its reactivity. Lead-like
  compounds have molecular weight between 300 and 350 Daltons and LogP
  between -1 and 3.5. Goldilocks compounds are lead-like compounds with
  LogP values further restricted to between 2 and 3.

  If `reload = True` and `data_dir` (`save_dir`) is specified, the loader
  will attempt to load the raw dataset (featurized dataset) from disk.
  Otherwise, the dataset will be downloaded from the DeepChem AWS bucket.

  For more information on ZINC15, please see [1]_ and
  https://zinc15.docking.org/.

  Parameters
  ----------
  size : str (default '250K')
    Size of dataset to download. Currently only '250K' is supported.
  format : str (default '2D')
    Format of data to download. 2D SMILES strings or 3D SDF files.
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
  dataset_size : str (default '250K')
    Number of compounds to download; '250K', '1M', '10M', or '270M'.
  dataset_dimension : str (default '2D')
    SMILES strings (2D) or 3D SDF files; '2D' or '3D'
  test_run : bool (default False)
    Flag to indicate tests, if True dataset is not downloaded.

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

  Notes
  -----
  The total ZINC dataset with SMILES strings contains hundreds of millions
  of compounds and is over 100GB! ZINC250K is recommended for experimentation.
  The full set of 270M goldilocks compounds is 23GB.

  References
  ----------
  .. [1] Sterling and Irwin. J. Chem. Inf. Model, 2015 http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559.

  Examples
  --------
  >>> import deepchem as dc
  >>> tasks, datasets, transformers = dc.molnet.load_zinc15(test_run=True)
  >>> train_dataset, val_dataset, test_dataset = datasets
  >>> n_tasks = len(tasks)
  >>> n_features = train_dataset.X.shape[1]
  >>> model = dc.models.MultitaskRegressor(n_tasks, n_features)

  """

  # Featurize zinc15
  logger.info("About to featurize zinc15.")
  my_tasks = ['mwt', 'logp', 'reactive']  # machine learning targets

  if test_run:
    ds = deepchem.data.NumpyDataset(np.zeros((10, 1)))
    return my_tasks, (ds, ds, ds), []

  # Raise warnings and list other available options
  if dataset_size not in ['250K', '1M', '10M', '270M']:
    raise ValueError("""
      Only '250K', '1M', '10M', and '270M' are supported for dataset_size.
      """)
  if dataset_dimension != '2D':
    raise ValueError("""
      Currently, only '2D' is supported for dataset_dimension.
      """)
  if dataset_size == '270M':
    answer = ''
    while answer not in ['y', 'n']:
      answer = input("""You're about to download 270M SMILES strings.
        This dataset is 23GB. Are you sure you want to continue? (Y/N)"""
                    ).lower()
    if answer == 'n':
      raise ValueError('Choose a smaller dataset_size.')

  dataset_filename = 'zinc15_' + dataset_size + '_' + dataset_dimension + '.tar.gz'

  zinc15_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/" + dataset_filename

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
    dataset_file = os.path.join(data_dir, dataset_filename)

    if not os.path.exists(dataset_file):
      deepchem.utils.data_utils.download_url(url=zinc15_URL, dest_dir=data_dir)

    deepchem.utils.data_utils.untargz_file(
        os.path.join(data_dir, dataset_filename), data_dir)
    dataset_file = 'zinc15_' + dataset_size + '_' + dataset_dimension + '.csv'

    loader = deepchem.data.CSVLoader(
        tasks=my_tasks,
        feature_field="smiles",
        id_field='zinc_id',
        featurizer=featurizer)

  # Featurize dataset
  dataset = loader.create_dataset(os.path.join(data_dir, dataset_file))

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


if __name__ == "__main__":
  import doctest
  doctest.testmod()
