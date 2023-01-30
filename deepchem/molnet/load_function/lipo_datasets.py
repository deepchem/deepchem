"""
Lipophilicity dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

LIPO_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
LIPO_TASKS = ['exp']


class _LipoLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "Lipophilicity.csv")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=LIPO_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_lipo(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Lipophilicity dataset

    Lipophilicity is an important feature of drug molecules that affects both
    membrane permeability and solubility. The lipophilicity dataset, curated
    from ChEMBL database, provides experimental results of octanol/water
    distribution coefficient (logD at pH 7.4) of 4200 compounds.

    Scaffold splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles" - SMILES representation of the molecular structure
    - "exp" - Measured octanol/water distribution coefficient (logD) of the
        compound, used as label

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
    .. [1] Hersey, A. ChEMBL Deposited Data Set - AZ dataset; 2015.
        https://doi.org/10.6019/chembl3301361
    """
    loader = _LipoLoader(featurizer, splitter, transformers, LIPO_TASKS,
                         data_dir, save_dir, **kwargs)
    return loader.load_dataset('lipo', reload)
