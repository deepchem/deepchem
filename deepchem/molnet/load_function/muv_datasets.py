"""
MUV dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

MUV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"
MUV_TASKS = sorted([
    'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
    'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
    'MUV-652', 'MUV-466', 'MUV-832'
])


class _MuvLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "muv.csv.gz")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=MUV_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_muv(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load MUV dataset

    The Maximum Unbiased Validation (MUV) group is a benchmark dataset selected
    from PubChem BioAssay by applying a refined nearest neighbor analysis.

    The MUV dataset contains 17 challenging tasks for around 90 thousand
    compounds and is specifically designed for validation of virtual screening
    techniques.

    Scaffold splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "mol_id" - PubChem CID of the compound
    - "smiles" - SMILES representation of the molecular structure
    - "MUV-XXX" - Measured results (Active/Inactive) for bioassays

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
    .. [1] Rohrer, Sebastian G., and Knut Baumann. "Maximum unbiased validation
        (MUV) data sets for virtual screening based on PubChem bioactivity data."
        Journal of chemical information and modeling 49.2 (2009): 169-184.
    """
    loader = _MuvLoader(featurizer, splitter, transformers, MUV_TASKS, data_dir,
                        save_dir, **kwargs)
    return loader.load_dataset('muv', reload)
