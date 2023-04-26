"""
ChEMBL dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from deepchem.molnet.load_function.chembl_tasks import chembl_tasks

from typing import List, Optional, Tuple, Union

CHEMBL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_%s.csv.gz"


class _ChemblLoader(_MolnetLoader):

    def __init__(self, *args, set: str, **kwargs):
        super(_ChemblLoader, self).__init__(*args, **kwargs)
        self.set = set

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir,
                                    "chembl_%s.csv.gz" % self.set)
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=CHEMBL_URL % self.set,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_chembl(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    set: str = "5thresh",
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load the ChEMBL dataset.

    This dataset is based on release 22.1 of the data from https://www.ebi.ac.uk/chembl/.
    Two subsets of the data are available, depending on the "set" argument.  "sparse"
    is a large dataset with 244,245 compounds.  As the name suggests, the data is
    extremely sparse, with most compounds having activity data for only one target.
    "5thresh" is a much smaller set (23,871 compounds) that includes only compounds
    with activity data for at least five targets.

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
    set: str
        the subset to load, either "sparse" or "5thresh"
    reload: bool
        if True, the first call for a particular featurizer and splitter will cache
        the datasets to disk, and subsequent calls will reload the cached datasets.
    data_dir: str
        a directory to save the raw data in
    save_dir: str
        a directory to save the dataset in
    """
    if set not in ("5thresh", "sparse"):
        raise ValueError("set must be either '5thresh' or 'sparse'")
    loader = _ChemblLoader(featurizer,
                           splitter,
                           transformers,
                           chembl_tasks,
                           data_dir,
                           save_dir,
                           set=set,
                           **kwargs)
    return loader.load_dataset('chembl-%s' % set, reload)
