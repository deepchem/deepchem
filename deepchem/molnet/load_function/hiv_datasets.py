"""
hiv dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

HIV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
HIV_TASKS = ["HIV_active"]


class _HIVLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "HIV.csv")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=HIV_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_hiv(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load HIV dataset

    The HIV dataset was introduced by the Drug Therapeutics
    Program (DTP) AIDS Antiviral Screen, which tested the ability
    to inhibit HIV replication for over 40,000 compounds.
    Screening results were evaluated and placed into three
    categories: confirmed inactive (CI),confirmed active (CA) and
    confirmed moderately active (CM). We further combine the
    latter two labels, making it a classification task between
    inactive (CI) and active (CA and CM).

    Scaffold splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles": SMILES representation of the molecular structure
    - "activity": Three-class labels for screening results: CI/CM/CA
    - "HIV_active": Binary labels for screening results: 1 (CA/CM) and 0 (CI)

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
    .. [1] AIDS Antiviral Screen Data.
        https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data
    """
    loader = _HIVLoader(featurizer, splitter, transformers, HIV_TASKS, data_dir,
                        save_dir, **kwargs)
    return loader.load_dataset('hiv', reload)
