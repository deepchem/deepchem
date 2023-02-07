"""
ZINC15 commercially-available compounds for virtual screening.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

ZINC15_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
ZINC15_TASKS = ['mwt', 'logp', 'reactive']


class _Zinc15Loader(_MolnetLoader):

    def __init__(self, *args, dataset_size: str, dataset_dimension: str,
                 **kwargs):
        super(_Zinc15Loader, self).__init__(*args, **kwargs)
        self.dataset_size = dataset_size
        self.dataset_dimension = dataset_dimension
        self.name = 'zinc15_' + dataset_size + '_' + dataset_dimension

    def create_dataset(self) -> Dataset:
        if self.dataset_size not in ['250K', '1M', '10M', '270M']:
            raise ValueError(
                "Only '250K', '1M', '10M', and '270M' are supported for dataset_size."
            )
        if self.dataset_dimension != '2D':
            raise ValueError(
                "Currently, only '2D' is supported for dataset_dimension.")
        if self.dataset_size == '270M':
            answer = ''
            while answer not in ['y', 'n']:
                answer = input("""You're about to download 270M SMILES strings.
                This dataset is 23GB. Are you sure you want to continue? (Y/N)"""
                              ).lower()
            if answer == 'n':
                raise ValueError('Choose a smaller dataset_size.')
        filename = self.name + '.csv'
        dataset_file = os.path.join(self.data_dir, filename)
        if not os.path.exists(dataset_file):
            compressed_file = self.name + '.tar.gz'
            if not os.path.exists(compressed_file):
                dc.utils.download_url(url=ZINC15_URL + compressed_file,
                                      dest_dir=self.data_dir)
            dc.utils.untargz_file(os.path.join(self.data_dir, compressed_file),
                                  self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   id_field="zinc_id",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_zinc15(
    featurizer: Union[dc.feat.Featurizer, str] = 'OneHot',
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    dataset_size: str = '250K',
    dataset_dimension: str = '2D',
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
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
    featurizer: Featurizer or str
        the featurizer to use for processing the data.  Alternatively you can pass
        one of the names from dc.molnet.featurizers as a shortcut.
    splitter: Splitter or str
        the splitter to use for splitting the data into training, validation, and
        test sets.  Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut.  If this is None, all the data
        will be included in a single dataset.
    transformers: list of TransformerGenerators or strings
        the Transformers to apply to the data.  Each one is specified by a
        TransformerGenerator or, as a shortcut, one of the names from
        dc.molnet.transformers.
    reload: bool
        if True, the first call for a particular featurizer and splitter will cache
        the datasets to disk, and subsequent calls will reload the cached datasets.
    data_dir: str
        a directory to save the raw data in
    save_dir: str
        a directory to save the dataset in
    size : str (default '250K')
        Size of dataset to download. '250K', '1M', '10M', and '270M' are supported.
    format : str (default '2D')
        Format of data to download. 2D SMILES strings or 3D SDF files.

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
    """
    loader = _Zinc15Loader(featurizer,
                           splitter,
                           transformers,
                           ZINC15_TASKS,
                           data_dir,
                           save_dir,
                           dataset_size=dataset_size,
                           dataset_dimension=dataset_dimension,
                           **kwargs)
    return loader.load_dataset(loader.name, reload)
