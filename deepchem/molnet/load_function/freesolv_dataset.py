"""
freesolv dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

FREESOLV_URL = 'https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz'
FREESOLV_TASKS = ['y']


class _FreesolvLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, 'freesolv.csv.gz')
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=FREESOLV_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field='smiles',
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file)


def load_freesolv(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.MATFeaturizer(),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Freesolv dataset

    The FreeSolv dataset is a collection of experimental and calculated hydration
    free energies for small molecules in water, along with their experiemental values.
    Here, we are using a modified version of the dataset with the molecule smile string
    and the corresponding experimental hydration free energies.


    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "mol" - SMILES representation of the molecular structure
    - "y" - Experimental hydration free energy

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
    .. [1] Łukasz Maziarka, et al. "Molecule Attention Transformer." NeurIPS 2019
        arXiv:2002.08264v1 [cs.LG].
    .. [2] Mobley DL, Guthrie JP. FreeSolv:
        a database of experimental and calculated hydration free energies, with input files.
        J Comput Aided Mol Des. 2014;28(7):711-720. doi:10.1007/s10822-014-9747-x
    """
    loader = _FreesolvLoader(featurizer, splitter, transformers, FREESOLV_TASKS,
                             data_dir, save_dir, **kwargs)
    return loader.load_dataset('freesolv', reload)
