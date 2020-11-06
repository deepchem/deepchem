"""
Platinum Adsorbtion structure for N and NO along with their formation energies
"""
import os
import logging
import deepchem
from deepchem.feat import Featurizer
from deepchem.splits.splitters import Splitter
from deepchem.molnet.defaults import get_defaults
import numpy as np

from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.data_utils.get_data_dir()
PLATINUM_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/platinum_adsorption.tar.gz"

# dict of accepted featurizers for this dataset
# modify the returned dicts for your dataset
DEFAULT_FEATURIZERS = get_defaults("feat")

# Names of supported featurizers
featurizers = ['LCNNFeaturizer']
DEFAULT_FEATURIZERS = {k: DEFAULT_FEATURIZERS[k] for k in featurizers}

# dict of accepted transformers
DEFAULT_TRANSFORMERS = get_defaults("trans")

# dict of accepted splitters
DEFAULT_SPLITTERS = get_defaults("splits")

# names of supported splitters
mydataset_splitters = ['RandomSplitter']
DEFAULT_SPLITTERS = {k: DEFAULT_SPLITTERS[k] for k in mydataset_splitters}


def load_Platinum_Adsorption(featurizer=DEFAULT_FEATURIZERS['LCNNFeaturizer'],
                             transformers: List = [],
                             splitter=DEFAULT_SPLITTERS['RandomSplitter'],
                             reload: bool = True,
                             data_dir: Optional[str] = None,
                             save_dir: Optional[str] = None,
                             featurizer_kwargs: Dict[str, object] = {},
                             splitter_kwargs: Dict[str, Any] = {
                                 'frac_train': 0.8,
                                 'frac_valid': 0.1,
                                 'frac_test': 0.1
                             },
                             transformer_kwargs: Dict[str, Any] = {},
                             **kwargs) -> Tuple[List, Optional[Tuple], List]:
  """Load mydataset.
  Contains 
  Parameters
  ----------
  featurizer : Featurizer (default LCNNFeaturizer)
    A featurizer that inherits from deepchem.feat.Featurizer.
  transformers : List[]
  Does'nt require any transformation
  splitter : Splitter (default RandomSplitter)
    A splitter that inherits from deepchem.splits.splitters.Splitter.
  reload : bool (default True)
    Try to reload dataset from disk if already downloaded. Save to disk
    after featurizing.
  data_dir : str, optional (default None)
    Path to datasets.
  save_dir : str, optional (default None)
    Path to featurized datasets.
  featurizer_kwargs : dict
    Specify parameters to featurizer, e.g. {"cutoff": 6.00}
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
  MLA style references for this dataset. The example is like this.
  Last, First et al. "Article title." Journal name, vol. #, no. #, year, pp. page range, DOI.
  ...[1] Lym, J et al. "Lattice Convolutional Neural Network Modeling of Adsorbate
         Coverage Effects"J. Phys. Chem. C 2019, 123, 18951−18959
  Examples
  --------
  >> import deepchem as dc
  >> feat_args = {"cutoff": np.around(6.00, 2), "input_file_path": os.path.join(data_path,'input.in') }

  >> tasks, datasets, transformers = load_Platinum_Adsorption(
      reload=True,
      data_dir=data_path,
      save_dir=data_path,
      featurizer_kwargs=feat_args)
  >> train_dataset, val_dataset, test_dataset = datasets

  """

  # Featurize mydataset
  logger.info("About to featurize Platinum Adsorption dataset.")
  my_tasks = ["Formation Energy"]  # machine learning targets

  # Get DeepChem data directory if needed
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if 'cutoff' not in featurizer_kwargs:
    raise TypeError("cutoff argument needs to be given")
  if 'input_file_path' not in featurizer_kwargs:
    raise TypeError("input_file_path argument needs to be given")

  #Download the data if does'nt exist
  dataset_file = os.path.join(data_dir, 'Platinum_Adsorption.json')
  if not os.path.exists(dataset_file):

    deepchem.utils.data_utils.download_url(url=PLATINUM_URL, dest_dir=data_dir)
    deepchem.utils.data_utils.untargz_file(
        os.path.join(data_dir, 'platinum_adsorption.tar.gz'), data_dir)

  # Check for str args to featurizer and splitter
  if issubclass(featurizer, Featurizer):
    featurizer = featurizer(**featurizer_kwargs)
  else:
    raise TypeError("featurizer must be a subclass of Featurizer.")

  if issubclass(splitter, Splitter):
    splitter = splitter()
  else:
    raise TypeError("splitter must be a subclass of Splitter.")

  # Reload from disk
  if reload:
    featurizer_name = str(featurizer.__class__.__name__)
    splitter_name = str(splitter.__class__.__name__)
    save_folder = os.path.join(save_dir, "Platinum_dataset", featurizer_name,
                               splitter_name)

    loaded, all_dataset, transformers = deepchem.utils.data_utils.load_dataset_from_disk(
        save_folder)
    if loaded:
      return my_tasks, all_dataset, transformers

  # First type of supported featurizers
  supported_featurizers: List[str] = ['LCNNFeaturizer'
                                     ]  # type: List[Featurizer]

  # If featurizer requires a non-CSV file format, load .tar.gz file
  if featurizer.__class__.__name__ in supported_featurizers:
    dataset_file = os.path.join(data_dir, 'Platinum_Adsorption.json')

    # Changer loader to match featurizer and data file type
    loader = deepchem.data.JsonLoader(
        tasks=my_tasks,
        feature_field="Structures",
        label_field="Formation Energy",
        featurizer=featurizer)

  else:
    raise TypeError("Wrong format of featurizers")

  # Featurize dataset
  dataset = loader.create_dataset(dataset_file)

  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset, **splitter_kwargs)

  # Initialize transformers
  transformers = [
      DEFAULT_TRANSFORMERS[t](dataset=dataset, **transformer_kwargs[t])
      if isinstance(t, str) else t(
          dataset=dataset, **transformer_kwargs[str(t.__class__.__name__)])
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
