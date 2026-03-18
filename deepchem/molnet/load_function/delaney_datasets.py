"""
Delaney dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

DELANEY_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
DELANEY_TASKS = ['measured log solubility in mols per litre']


class _DelaneyLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "delaney-processed.csv")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=DELANEY_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_delaney(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Delaney dataset

    The Delaney (ESOL) dataset a regression dataset containing structures and
    water solubility data for 1128 compounds. The dataset is widely used to
    validate machine learning models on estimating solubility directly from
    molecular structures (as encoded in SMILES strings).

    Scaffold splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "Compound ID" - Name of the compound
    - "smiles" - SMILES representation of the molecular structure
    - "measured log solubility in mols per litre" - Log-scale water solubility
        of the compound, used as label

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

    References
    ----------
    .. [1] Delaney, John S. "ESOL: estimating aqueous solubility directly from
        molecular structure." Journal of chemical information and computer
        sciences 44.3 (2004): 1000-1005.
    """
    loader = _DelaneyLoader(featurizer, splitter, transformers, DELANEY_TASKS,
                            data_dir, save_dir, **kwargs)
    return loader.load_dataset('delaney', reload)
