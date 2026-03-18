"""
Short docstring description of dataset.
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
MYDATASET_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/mydataset.tar.gz"
MYDATASET_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/mydataset.csv"

# dict of accepted featurizers for this dataset
# modify the returned dicts for your dataset
DEFAULT_FEATURIZERS = get_defaults("feat")

# Names of supported featurizers
mydataset_featurizers = ['CircularFingerprint', 'ConvMolFeaturizer']
DEFAULT_FEATURIZERS = {k: DEFAULT_FEATURIZERS[k] for k in mydataset_featurizers}

# dict of accepted transformers
DEFAULT_TRANSFORMERS = get_defaults("trans")

# dict of accepted splitters
DEFAULT_SPLITTERS = get_defaults("splits")

# names of supported splitters
mydataset_splitters = ['RandomSplitter', 'RandomStratifiedSplitter']
DEFAULT_SPLITTERS = {k: DEFAULT_SPLITTERS[k] for k in mydataset_splitters}


def load_mydataset(
        featurizer: Featurizer = DEFAULT_FEATURIZERS['CircularFingerprint'],
        transformers: List[Transformer] = [
            DEFAULT_TRANSFORMERS['NormalizationTransformer']
        ],
        splitter: Splitter = DEFAULT_SPLITTERS['RandomSplitter'],
        reload: bool = True,
        data_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        featurizer_kwargs: Dict[str, object] = {},
        splitter_kwargs: Dict[str, object] = {},
        transformer_kwargs: Dict[str, Dict[str, object]] = {},
        **kwargs) -> Tuple[List, Tuple, List]:
    """Load mydataset.

    This is a template for adding a function to load a dataset from
    MoleculeNet. Adjust the global variable URL strings, default parameters,
    default featurizers, transformers, and splitters, and variable names as
    needed. All available featurizers, transformers, and
    splitters are in the `DEFAULTS_X` global variables.

    If `reload = True` and `data_dir` (`save_dir`) is specified, the loader
    will attempt to load the raw dataset (featurized dataset) from disk.
    Otherwise, the dataset will be downloaded from the DeepChem AWS bucket.

    The dataset will be featurized with `featurizer` and separated into
    train/val/test sets according to `splitter`. Some transformers (e.g.
    `NormalizationTransformer`) must be initialized with a dataset.
    Set up kwargs to enable these transformations. Additional kwargs may
    be given for specific featurizers, transformers, and splitters.

    The load function must be modified with the appropriate DataLoaders
    for all supported featurizers for your dataset.

    Please refer to the MoleculeNet documentation for further information
    https://deepchem.readthedocs.io/en/latest/moleculenet.html.

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
    MLA style references for this dataset. The example is like this.
    Last, First et al. "Article title." Journal name, vol. #, no. #, year, pp. page range, DOI.
    ...[1] Wu, Zhenqin et al. "MoleculeNet: a benchmark for molecular machine learning."
        Chemical Science, vol. 9, 2018, pp. 513-530, 10.1039/c7sc02664a.

    Examples
    --------
    >> import deepchem as dc
    >> tasks, datasets, transformers = dc.molnet.load_tox21(reload=False)
    >> train_dataset, val_dataset, test_dataset = datasets
    >> n_tasks = len(tasks)
    >> n_features = train_dataset.get_data_shape()[0]
    >> model = dc.models.MultitaskClassifier(n_tasks, n_features)
    """

    # Warning message about this template
    raise ValueError("""
        This is a template function and it doesn't do anything!
        Use this function as a reference when implementing new
        loaders for MoleculeNet datasets.
        """)

    # Featurize mydataset
    logger.info("About to featurize mydataset.")
    my_tasks = ["task1", "task2", "task3"]  # machine learning targets

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
        save_folder = os.path.join(save_dir, "mydataset-featurized",
                                   featurizer_name, splitter_name)

        loaded, all_dataset, transformers = deepchem.utils.data_utils.load_dataset_from_disk(
            save_folder)
        if loaded:
            return my_tasks, all_dataset, transformers

    # First type of supported featurizers
    supported_featurizers = []  # type: List[Featurizer]

    # If featurizer requires a non-CSV file format, load .tar.gz file
    if featurizer in supported_featurizers:
        dataset_file = os.path.join(data_dir, 'mydataset.filetype')

        if not os.path.exists(dataset_file):
            deepchem.utils.data_utils.download_url(url=MYDATASET_URL,
                                                   dest_dir=data_dir)
            deepchem.utils.data_utils.untargz_file(
                os.path.join(data_dir, 'mydataset.tar.gz'), data_dir)

        # Changer loader to match featurizer and data file type
        loader = deepchem.data.DataLoader(
            tasks=my_tasks,
            id_field="id",  # column name holding sample identifier
            featurizer=featurizer)

    else:  # only load CSV file
        dataset_file = os.path.join(data_dir, "mydataset.csv")
        if not os.path.exists(dataset_file):
            deepchem.utils.data_utils.download_url(url=MYDATASET_CSV_URL,
                                                   dest_dir=data_dir)

        loader = deepchem.data.CSVLoader(tasks=my_tasks,
                                         smiles_field="smiles",
                                         featurizer=featurizer)

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
        deepchem.utils.data_utils.save_dataset_to_disk(save_folder,
                                                       train_dataset,
                                                       valid_dataset,
                                                       test_dataset,
                                                       transformers)

    return my_tasks, (train_dataset, valid_dataset, test_dataset), transformers
