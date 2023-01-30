"""
Perovskite crystal structures and formation energies.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

PEROVSKITE_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/perovskite.tar.gz'
PEROVSKITE_TASKS = ['formation_energy']


class _PerovskiteLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, 'perovskite.json')
        targz_file = os.path.join(self.data_dir, 'perovskite.tar.gz')
        if not os.path.exists(dataset_file):
            if not os.path.exists(targz_file):
                dc.utils.data_utils.download_url(url=PEROVSKITE_URL,
                                                 dest_dir=self.data_dir)
            dc.utils.data_utils.untargz_file(targz_file, self.data_dir)
        loader = dc.data.JsonLoader(tasks=self.tasks,
                                    feature_field="structure",
                                    label_field="formation_energy",
                                    featurizer=self.featurizer)
        return loader.create_dataset(dataset_file)


def load_perovskite(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.DummyFeaturizer(),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load perovskite dataset.

    Contains 18928 perovskite structures and their formation energies.
    In benchmark studies, random forest models and crystal graph
    neural networks achieved mean average error of 0.23 and 0.05 eV/atom,
    respectively, during five-fold nested cross validation on this
    dataset.

    For more details on the dataset see [1]_. For more details
    on previous benchmarks for this dataset, see [2]_.

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
    .. [1] Castelli, I. et al. "New cubic perovskites for one- and two-photon water splitting
        using the computational materials repository." Energy Environ. Sci., (2012), 5,
        9034-9043 DOI: 10.1039/C2EE22341D.
    .. [2] Dunn, A. et al. "Benchmarking Materials Property Prediction Methods:
        The Matbench Test Set and Automatminer Reference Algorithm." https://arxiv.org/abs/2005.00707 (2020)

    Examples
    --------
    >>> import deepchem as dc
    >>> tasks, datasets, transformers = dc.molnet.load_perovskite()
    >>> train_dataset, val_dataset, test_dataset = datasets
    >>> model = dc.models.CGCNNModel(mode='regression', batch_size=32, learning_rate=0.001)

    """
    loader = _PerovskiteLoader(featurizer, splitter, transformers,
                               PEROVSKITE_TASKS, data_dir, save_dir, **kwargs)
    return loader.load_dataset('perovskite', reload)
