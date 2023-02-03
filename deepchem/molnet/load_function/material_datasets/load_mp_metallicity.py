"""
Metal vs non-metal classification for inorganic crystals from Materials Project.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

MPMETAL_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/mp_is_metal.tar.gz'
MPMETAL_TASKS = ['is_metal']


class _MPMetallicityLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, 'mp_is_metal.json')
        targz_file = os.path.join(self.data_dir, 'mp_is_metal.tar.gz')
        if not os.path.exists(dataset_file):
            if not os.path.exists(targz_file):
                dc.utils.data_utils.download_url(url=MPMETAL_URL,
                                                 dest_dir=self.data_dir)
            dc.utils.data_utils.untargz_file(targz_file, self.data_dir)
        loader = dc.data.JsonLoader(tasks=self.tasks,
                                    feature_field="structure",
                                    label_field="is_metal",
                                    featurizer=self.featurizer)
        return loader.create_dataset(dataset_file)


def load_mp_metallicity(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.SineCoulombMatrix(),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load mp formation energy dataset.

    Contains 106113 inorganic crystal structures from the Materials
    Project database labeled as metals or nonmetals. In benchmark
    studies, random forest models achieved a mean ROC-AUC of
    0.9 during five-folded nested cross validation on this
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
    .. [1] A. Jain*, S.P. Ong*, et al. (*=equal contributions) The Materials Project:
        A materials genome approach to accelerating materials innovation APL Materials,
        2013, 1(1), 011002. doi:10.1063/1.4812323 (2013).
    .. [2] Dunn, A. et al. "Benchmarking Materials Property Prediction Methods: The Matbench
        Test Set and Automatminer Reference Algorithm." https://arxiv.org/abs/2005.00707 (2020)

    Examples
    --------
    >>>
    >> import deepchem as dc
    >> tasks, datasets, transformers = dc.molnet.load_mp_metallicity()
    >> train_dataset, val_dataset, test_dataset = datasets
    >> n_tasks = len(tasks)
    >> n_features = train_dataset.get_data_shape()[0]
    >> model = dc.models.MultitaskRegressor(n_tasks, n_features)

    """
    loader = _MPMetallicityLoader(featurizer, splitter, transformers,
                                  MPMETAL_TASKS, data_dir, save_dir, **kwargs)
    return loader.load_dataset('mp-metallicity', reload)
