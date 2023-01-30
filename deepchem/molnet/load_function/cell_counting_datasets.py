"""
Cell Counting Dataset.

Loads the cell counting dataset from
http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html. Labels aren't
available for this dataset, so only raw images are provided.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

CELL_COUNTING_URL = 'http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip'
CELL_COUNTING_TASKS: List[str] = []


class _CellCountingLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "cells.zip")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=CELL_COUNTING_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.ImageLoader()
        return loader.featurize(dataset_file)


def load_cell_counting(
    splitter: Union[dc.splits.Splitter, str, None] = None,
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Cell Counting dataset.

    Loads the cell counting dataset from http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html.

    Parameters
    ----------
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
    """
    featurizer = dc.feat.UserDefinedFeaturizer([])  # Not actually used
    loader = _CellCountingLoader(featurizer, splitter, transformers,
                                 CELL_COUNTING_TASKS, data_dir, save_dir,
                                 **kwargs)
    return loader.load_dataset('cell_counting', reload)
