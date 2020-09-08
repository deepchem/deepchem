"""
Loading Kamlet-Taft (KT) parameter dataset 
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

DEFAULT_DIR = deepchem.utils.get_data_dir()
KTDATASET_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KTparameterDataset.zip"
KTDATASET_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/KTparameterDataset.csv"

# dict of accepted featurizers for the KT dataset
DEFAULT_FEATURIZERS = get_defaults("feat")

# Names of supported featurizers
# ktdataset_featurizers = ["RawFeaturizer", "SmilesToImage", "SmilesToSeq", "WeaveFeaturizer"]
ktdataset_featurizers = ['ECFP', 'GraphConv', 'Weave', 'Raw', 'AdjacencyConv']
DEFAULT_FEATURIZERS = {k: DEFAULT_FEATURIZERS[k] for k in ktdataset_featurizers}

# dict of accepted transformers
DEFAULT_TRANSFORMERS = get_defaults("trans")

# dict of accepted splitters
DEFAULT_SPLITTERS = get_defaults("splits")

# names of supported splitters
ktdataset_splitters = [
    'index', 'random', 'scaffold', 'butina', 'task', 'stratified'
]
DEFAULT_SPLITTERS = {k: DEFAULT_SPLITTERS[k] for k in ktdataset_splitters}


def load_kt_dataset(
    featurizer: Featurizer = DEFAULT_FEATURIZERS['RawFeaturizer'],
    transformers: List[Transformer] = [
        DEFAULT_TRANSFORMERS['PowerTransformer']
    ],
    splitter: Splitter = DEFAULT_SPLITTERS['RandomSplitter'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    featurizer_kwargs: Dict[str, object] = {},
    splitter_kwargs: Dict[str, object] = {},
    transformer_kwargs: Dict[str, Dict[str, object]] = {},
    **kwargs) -> (Tuple[List, Tuple, List]):

"""

  KT parameters are scales to measure and quantify the Lewis acidity and basicity of molecules. The parameters are obtained through Nuclear Magnetic Resonance (NMR) spectra. Random splitting is recommended for this dataset.
  The raw data csv file contains columns below:
  Please refer to https://arxiv.org/pdf/2008.08078 (2020) for which the dataset was hand curated. Benchmark training is also present in this work. Accuracy levels (rmse) reported here are of the order 0.01  for alpha and beta on unseen data. 
   References
  ----------
  [1] Marcus, Yizhak. "The properties of organic liquids that are relevant to their use as solvating solvents." Chemical Society Reviews 22, no. 6 (1993): 409-416.


  Loading ktdataset.

  - "csmiles" - canonical SMILES representation of the molecular structure
  - "cid" - PubChem CID of molecules
  - "alpha" - KT paramter that quantifies the acidity of a molecule
  - "beta" - KT paramter that quantifies the basicity of a molecule
  
  Parameters
  ----------
  featurizer : {List of allowed featurizers for this dataset}
    A featurizer that inherits from deepchem.feat.Featurizer.
  transformers : List{List of allowed transformers for this dataset}
    A transformer that inherits from deepchem.trans.Transformer.
  splitter : {List of allowed splitters for this dataset}
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
  MLA style references for the KT parameter dataset. 
    [1] Marcus, Yizhak. "The properties of organic liquids that are relevant to their use as solvating solvents." Chemical Society Reviews 22.6 (1993): 409-416.
  Examples
  --------
  >> import deepchem as dc
  >> tasks, datasets, transformers = dc.molnet.load_kt_dataset(reload=False)
  >> train_dataset, val_dataset, test_dataset = datasets
  >> n_tasks = len(tasks)
  >> n_features = train_dataset.get_data_shape()[0]
  >> model = dc.models.MultitaskClassifier(n_tasks, n_features)
"""


  
  

  # Featurize ktdataset
  logger.info("About to featurize ktdataset.")
  kt_tasks = ["alpha", "beta"]  # machine learning targets

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
    save_folder = os.path.join(save_dir, "ktdataset-featurized",
                               featurizer_name, splitter_name)

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return kt_tasks, all_dataset, transformers

  # First type of supported featurizers
  supported_featurizers = ['ECFP', 'GraphConv', 'Weave', 'Raw', 'AdjacencyConv']  # type: List[Featurizer]

  # If featurizer requires a non-CSV file format, load .tar.gz file
  if featurizer in supported_featurizers:
    dataset_file = os.path.join(data_dir, 'KTparameterDataset.csv')

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=KTDATASET_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'ktdataset.tar.gz'), data_dir)

    # Changer loader to match featurizer and data file type
    loader = deepchem.data.DataLoader(
        tasks=kt_tasks,
        id_field="cid",  # column name holding sample identifier
        featurizer=featurizer)

  else:  # only load CSV file
    dataset_file = os.path.join(data_dir, "ktdataset.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=KTDATASET_CSV_URL, dest_dir=data_dir)

    loader = deepchem.data.CSVLoader(
        tasks=kt_tasks, smiles_field="csmiles", featurizer=featurizer)

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
    deepchem.utils.save.save_dataset_to_disk(
        save_folder, train_dataset, valid_dataset, test_dataset, transformers)

  return kt_tasks, (train_dataset, valid_dataset, test_dataset), transformers

