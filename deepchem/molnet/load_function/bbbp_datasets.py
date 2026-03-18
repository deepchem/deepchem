"""
Blood-Brain Barrier Penetration dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
BBBP_TASKS = ["p_np"]


class _BBBPLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "BBBP.csv")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBP_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_bbbp(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load BBBP dataset

    The blood-brain barrier penetration (BBBP) dataset is designed for the
    modeling and prediction of barrier permeability. As a membrane separating
    circulating blood and brain extracellular fluid, the blood-brain barrier
    blocks most drugs, hormones and neurotransmitters. Thus penetration of the
    barrier forms a long-standing issue in development of drugs targeting
    central nervous system.

    This dataset includes binary labels for over 2000 compounds on their
    permeability properties.

    Scaffold splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "name" - Name of the compound
    - "smiles" - SMILES representation of the molecular structure
    - "p_np" - Binary labels for penetration/non-penetration

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
    .. [1] Martins, Ines Filipa, et al. "A Bayesian approach to in silico
        blood-brain barrier penetration modeling." Journal of chemical
        information and modeling 52.6 (2012): 1686-1697.
    """
    loader = _BBBPLoader(featurizer, splitter, transformers, BBBP_TASKS,
                         data_dir, save_dir, **kwargs)
    return loader.load_dataset('bbbp', reload)
