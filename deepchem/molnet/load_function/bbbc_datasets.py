"""
BBBC Dataset loader.

This file contains image loaders for the BBBC dataset collection (https://data.broadinstitute.org/bbbc/image_sets.html).
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

BBBC1_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_images_tif.zip'
BBBC1_LABEL_URL = 'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_counts.txt'
BBBC1_TASKS = ["cell-count"]

BBBC2_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_images.zip'
BBBC2_LABEL_URL = 'https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_counts.txt'
BBBC2_TASKS = ["cell-count"]


class _BBBC001Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "BBBC001_v1_images_tif.zip")
        labels_file = os.path.join(self.data_dir, "BBBC001_v1_counts.txt")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBC1_IMAGE_URL,
                                             dest_dir=self.data_dir)
        if not os.path.exists(labels_file):
            dc.utils.data_utils.download_url(url=BBBC1_LABEL_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.ImageLoader()
        return loader.create_dataset(dataset_file, in_memory=False)


def load_bbbc001(
    splitter: Union[dc.splits.Splitter, str, None] = 'index',
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load BBBC001 dataset

    This dataset contains 6 images of human HT29 colon cancer cells. The task is
    to learn to predict the cell counts in these images. This dataset is too small
    to serve to train algorithms, but might serve as a good test dataset.
    https://data.broadinstitute.org/bbbc/BBBC001/

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
    loader = _BBBC001Loader(featurizer, splitter, transformers, BBBC1_TASKS,
                            data_dir, save_dir, **kwargs)
    return loader.load_dataset('bbbc001', reload)


class _BBBC002Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "BBBC002_v1_images.zip")
        labels_file = os.path.join(self.data_dir, "BBBC002_v1_counts.txt.txt")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBC2_IMAGE_URL,
                                             dest_dir=self.data_dir)
        if not os.path.exists(labels_file):
            dc.utils.data_utils.download_url(url=BBBC2_LABEL_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.ImageLoader()
        return loader.create_dataset(dataset_file, in_memory=False)


def load_bbbc002(
    splitter: Union[dc.splits.Splitter, str, None] = 'index',
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load BBBC002 dataset

    This dataset contains data corresponding to 5 samples of Drosophilia Kc167
    cells. There are 10 fields of view for each sample, each an image of size
    512x512. Ground truth labels contain cell counts for this dataset. Full
    details about this dataset are present at
    https://data.broadinstitute.org/bbbc/BBBC002/.

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
    loader = _BBBC002Loader(featurizer, splitter, transformers, BBBC2_TASKS,
                            data_dir, save_dir, **kwargs)
    return loader.load_dataset('bbbc002', reload)
