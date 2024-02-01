"""
BBBC Dataset loader.

This file contains image loaders for the BBBC dataset collection (https://data.broadinstitute.org/bbbc/image_sets.html).
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union
import zipfile
import numpy as np
import pandas as pd

BBBC1_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_images_tif.zip'
BBBC1_LABEL_URL = 'https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_counts.txt'
BBBC1_TASKS = ["cell-count"]

BBBC2_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_images.zip'
BBBC2_LABEL_URL = 'https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_counts.txt'
BBBC2_TASKS = ["cell-count"]

BBBC3_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC003/BBBC003_v1_images.zip'
BBBC3_LABEL_URL = 'https://data.broadinstitute.org/bbbc/BBBC003/BBBC003_v1_counts.txt'
BBBC3_FOREGROUND_URL = 'https://data.broadinstitute.org/bbbc/BBBC003/BBBC003_v1_foreground.zip'
BBBC3_TASKS = ["cell-count"]

BBBC4_TASKS = ["cell-count"]

BBBC5_IMAGE_URL = 'https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip'
BBBC5_FOREGROUND_URL = 'https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip'
BBBC5_TASKS = ["cell-count"]


class _BBBC001_Loader(_MolnetLoader):
    """BBBC001 cell count dataset loader"""

    def create_dataset(self) -> Dataset:
        """Creates a dataset from BBBC001 images and cell counts as labels"""
        dataset_file = os.path.join(self.data_dir, "BBBC001_v1_images_tif.zip")
        labels_file = os.path.join(self.data_dir, "BBBC001_v1_counts.txt")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBC1_IMAGE_URL,
                                             dest_dir=self.data_dir)
        if not os.path.exists(labels_file):
            dc.utils.data_utils.download_url(url=BBBC1_LABEL_URL,
                                             dest_dir=self.data_dir)
        labels_table = pd.read_csv(labels_file, delimiter="\t")
        labels = np.mean(
            [labels_table["manual count #1"], labels_table["manual count #2"]],
            axis=0,
            dtype=int)

        loader = dc.data.ImageLoader()
        return loader.create_dataset(inputs=(dataset_file, labels),
                                     in_memory=False)


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
    loader = _BBBC001_Loader(featurizer, splitter, transformers, BBBC1_TASKS,
                             data_dir, save_dir, **kwargs)
    return loader.load_dataset('bbbc001', reload)


class _BBBC002_Loader(_MolnetLoader):
    """BBBC002 cell count dataset loader"""

    def create_dataset(self) -> Dataset:
        """Creates a dataset from BBBC002 images and cell counts as labels"""
        dataset_file = os.path.join(self.data_dir, "BBBC002_v1_images.zip")
        labels_file = os.path.join(self.data_dir, "BBBC002_v1_counts.txt")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBC2_IMAGE_URL,
                                             dest_dir=self.data_dir)
        if not os.path.exists(labels_file):
            dc.utils.data_utils.download_url(url=BBBC2_LABEL_URL,
                                             dest_dir=self.data_dir)

        labels_table = pd.read_csv(labels_file, delimiter="\t")
        labels = np.mean([
            labels_table["human counter 1 (Robert Lindquist)"],
            labels_table["human counter #2 (Joohan Chang)"]
        ],
                         axis=0,
                         dtype=int)

        loader = dc.data.ImageLoader()
        return loader.create_dataset(inputs=(dataset_file, labels),
                                     in_memory=False)


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
    loader = _BBBC002_Loader(featurizer, splitter, transformers, BBBC2_TASKS,
                             data_dir, save_dir, **kwargs)
    return loader.load_dataset('bbbc002', reload)


class _BBBC003_Segmentation_Loader(_MolnetLoader):
    """BBBC003 segmentation mask dataset loader"""

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "BBBC003_v1_images.zip")
        foreground_file = os.path.join(self.data_dir,
                                       "BBBC003_v1_foreground.zip")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBC3_IMAGE_URL,
                                             dest_dir=self.data_dir)
        if not os.path.exists(foreground_file):
            dc.utils.data_utils.download_url(url=BBBC3_FOREGROUND_URL,
                                             dest_dir=self.data_dir)

        loader = dc.data.ImageLoader(sorting=True)
        return loader.create_dataset(inputs=(dataset_file, foreground_file),
                                     in_memory=False)


class _BBBC003_Loader(_MolnetLoader):
    """BBBC003 cell count dataset loader"""

    def create_dataset(self):
        dataset_file = os.path.join(self.data_dir, "BBBC003_v1_images.zip")
        labels_file = os.path.join(self.data_dir, "BBBC003_v1_counts.txt")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBC3_IMAGE_URL,
                                             dest_dir=self.data_dir)
        if not os.path.exists(labels_file):
            dc.utils.data_utils.download_url(url=BBBC3_LABEL_URL,
                                             dest_dir=self.data_dir)

        labels = pd.read_csv(labels_file, delimiter="\t")
        lbx = labels.sort_values("Image")["manual count #1"].values

        loader = dc.data.ImageLoader(sorting=True)
        return loader.create_dataset(inputs=(dataset_file, lbx),
                                     in_memory=False)


def load_bbbc003(
    load_segmentation_mask: bool = False,
    splitter: Union[dc.splits.Splitter, str, None] = 'index',
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load BBBC003 dataset

    This dataset contains data corresponding to 15 samples of Mouse embryos with DIC.
    Each image is of size 640x480. Ground truth labels contain cell counts and
    segmentation masks for this dataset. Full details about this dataset are present at
    https://data.broadinstitute.org/bbbc/BBBC003/.

    Parameters
    ----------
    load_segmentation_mask: bool
        if True, the dataset will contain segmentation masks as labels. Otherwise,
        the dataset will contain cell counts as labels.
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

    Examples
    --------
    Importing necessary modules

    >>> import deepchem as dc
    >>> import numpy as np

    We can load the BBBC003 dataset with 2 types of labels: segmentation masks and
    cell counts. We will first load the dataset with cell counts as labels.

    >>> loader = dc.molnet.load_bbbc003(load_segmentation_mask=False)
    >>> tasks, dataset, transformers = loader
    >>> train, val, test = dataset

    We now have a dataset with 15 samples, each with 300 cells. The images are of
    size 640x480. The labels are cell counts. We can verify this as follows:

    >>> train.X.shape
    (12,)
    >>> train.y.shape
    (12,)

    We will now load the dataset with segmentation masks as labels.

    >>> loader = dc.molnet.load_bbbc003(load_segmentation_mask=True)
    >>> tasks, dataset, transformers = loader
    >>> train, val, test = dataset

    We now have a dataset with 15 samples, each with 300 cells. The images are of
    size 640x480. The labels are segmentation masks. We can verify this as follows:

    >>> print(train.X.shape)
    (12,)
    >>> print(train.y.shape)
    (12,)

    Note: The image labelled '7_19_M2E15.tif' is transposed to 480x640 in the source file along with it's
    segementation mask. To match it with the other images, we need to transpose it back to 640x480.

    This image is found at index 6 in the train dataset (Assuming no shuffling has taken place).

    First, we load the dataset as usual and split it into X, y, w and ids. Here, X is the list
    of input images, y is the list of labels, w is the list of weights and ids is the list of
    IDs for each sample.

    >>> train_x, train_y, train_w, train_ids = train.X, train.y, train.w, train.ids

    We can now transpose the image at index 6 in the input data (train_x):
    >>> train_x[6] = train_x[6].T

    We can now verify that the image is of size 640x480:
    >>> print(train_x[6].shape)
    (640, 480)

    This is also seen in the segmentation mask with the same filename and index, in which
    case, we transpose the label (train_y) instead of the input data:

    >>> train_y[6] = train_y[6].T

    We can now verify that the image is of size 640x480:
    >>> train_y[6].shape
    (640, 480)
    """
    featurizer = dc.feat.UserDefinedFeaturizer([])  # Not actually used
    loader: _MolnetLoader
    if load_segmentation_mask:
        loader = _BBBC003_Segmentation_Loader(featurizer, splitter,
                                              transformers, BBBC3_TASKS,
                                              data_dir, save_dir, **kwargs)
    else:
        loader = _BBBC003_Loader(featurizer, splitter, transformers,
                                 BBBC3_TASKS, data_dir, save_dir, **kwargs)

    return loader.load_dataset('bbbc003', reload)


class _BBBC004_Segmentation_Loader(_MolnetLoader):
    """BBBC004 segmentation mask dataset loader"""

    def __init__(self, overlap_probability, featurizer, splitter, transformers,
                 BBBC4_TASKS, data_dir, save_dir, **kwargs):
        overlap_dict = {
            0.0: "000",
            0.15: "015",
            0.3: "030",
            0.45: "045",
            0.6: "060"
        }
        if overlap_probability not in overlap_dict.keys():
            raise ValueError(
                f"Overlap_probability must be one of {overlap_dict.keys()}, got {overlap_probability}"
            )
        else:
            self.overlap_probability = overlap_dict[overlap_probability]
            self.BBBC4_IMAGE_URL = f"https://data.broadinstitute.org/bbbc/BBBC004/BBBC004_v1_{self.overlap_probability}_images.zip"
            self.BBBC4_FOREGROUND_URL = f"https://data.broadinstitute.org/bbbc/BBBC004/BBBC004_v1_{self.overlap_probability}_foreground.zip"

        super(_BBBC004_Segmentation_Loader,
              self).__init__(featurizer, splitter, transformers, BBBC4_TASKS,
                             data_dir, save_dir, **kwargs)

    def create_dataset(self) -> Dataset:
        """Creates a dataset from BBBC004 images and segmentation masks as labels"""

        dataset_file = os.path.join(
            self.data_dir, f"BBBC004_v1_{self.overlap_probability}_images.zip")
        foreground_file = os.path.join(
            self.data_dir,
            f"BBBC004_v1_{self.overlap_probability}_foreground.zip")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=self.BBBC4_IMAGE_URL,
                                             dest_dir=self.data_dir)
        if not os.path.exists(foreground_file):
            dc.utils.data_utils.download_url(url=self.BBBC4_FOREGROUND_URL,
                                             dest_dir=self.data_dir)

        loader = dc.data.ImageLoader(sorting=False)
        return loader.create_dataset(inputs=(dataset_file, foreground_file),
                                     in_memory=False)


class _BBBC004_Loader(_MolnetLoader):
    """BBBC004 cell count dataset loader"""

    def __init__(self, overlap_probability, featurizer, splitter, transformers,
                 BBBC4_TASKS, data_dir, save_dir, **kwargs):
        overlap_dict = {
            0.0: "000",
            0.15: "015",
            0.3: "030",
            0.45: "045",
            0.6: "060"
        }
        if overlap_probability not in overlap_dict.keys():
            raise ValueError(
                f"Overlap_probability must be one of {overlap_dict.keys()}, got {overlap_probability}"
            )
        else:
            self.overlap_probability = overlap_dict[overlap_probability]
            self.BBBC4_IMAGE_URL = f"https://data.broadinstitute.org/bbbc/BBBC004/BBBC004_v1_{self.overlap_probability}_images.zip"
            self.BBBC4_FOREGROUND_URL = f"https://data.broadinstitute.org/bbbc/BBBC004/BBBC004_v1_{self.overlap_probability}_foreground.zip"

        super(_BBBC004_Loader,
              self).__init__(featurizer, splitter, transformers, BBBC4_TASKS,
                             data_dir, save_dir, **kwargs)

    def create_dataset(self) -> Dataset:
        """Creates a dataset from BBBC004 images and cell counts as labels"""

        dataset_file = os.path.join(
            self.data_dir, f"BBBC004_v1_{self.overlap_probability}_images.zip")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=self.BBBC4_IMAGE_URL,
                                             dest_dir=self.data_dir)
        labels = np.full(20, 300, dtype=int)

        loader = dc.data.ImageLoader(sorting=False)
        return loader.create_dataset(inputs=(dataset_file, labels),
                                     in_memory=False)


def load_bbbc004(
    overlap_probability: float = 0.0,
    load_segmentation_mask: bool = False,
    splitter: Union[dc.splits.Splitter, str, None] = 'index',
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load BBBC004 dataset

    This dataset contains data corresponding to 20 samples of synthetically generated
    fluorescent cell population images. There are 300 cells in each sample, each an image
    of size 950x950. Ground truth labels contain cell counts and segmentation masks for
    this dataset. Full details about this dataset are present at
    https://data.broadinstitute.org/bbbc/BBBC004/.

    Parameters
    ----------
    overlap_probability: float from list {0.0, 0.15, 0.3, 0.45, 0.6}
        the overlap probability of the synthetic cells in the images
    load_segmentation_mask: bool
        if True, the dataset will contain segmentation masks as labels. Otherwise,
        the dataset will contain cell counts as labels.
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

    Examples
    --------
    Importing necessary modules

    >>> import deepchem as dc
    >>> import numpy as np

    We can load the BBBC004 dataset with 2 types of labels: segmentation masks and
    cell counts. We will first load the dataset with cell counts as labels.

    >>> loader = dc.molnet.load_bbbc004(overlap_probability=0.0, load_segmentation_mask=False)
    >>> tasks, dataset, transformers = loader
    >>> train, val, test = dataset

    We now have a dataset with 20 samples, each with 300 cells. The images are of
    size 950x950. The labels are cell counts. We can verify this as follows:

    >>> train.X.shape
    (16, 950, 950)
    >>> train.y.shape
    (16,)

    We will now load the dataset with segmentation masks as labels.

    >>> loader = dc.molnet.load_bbbc004(overlap_probability=0.0, load_segmentation_mask=True)
    >>> tasks, dataset, transformers = loader
    >>> train, val, test = dataset

    We now have a dataset with 20 samples, each with 300 cells. The images are of
    size 950x950. The labels are segmentation masks. We can verify this as follows:

    >>> train.X.shape
    (16, 950, 950)
    >>> train.y.shape
    (16, 950, 950, 3)
    """
    featurizer = dc.feat.UserDefinedFeaturizer([])  # Not actually used
    loader: _MolnetLoader
    if load_segmentation_mask:
        loader = _BBBC004_Segmentation_Loader(overlap_probability, featurizer,
                                              splitter, transformers,
                                              BBBC4_TASKS, data_dir, save_dir,
                                              **kwargs)
    else:
        loader = _BBBC004_Loader(overlap_probability, featurizer, splitter,
                                 transformers, BBBC4_TASKS, data_dir, save_dir,
                                 **kwargs)

    return loader.load_dataset('bbbc004', reload)


class _BBBC005_Loader(_MolnetLoader):
    """BBBC005 cell count dataset loader"""

    def create_dataset(self):
        dataset_file = os.path.join(self.data_dir, "BBBC005_v1_images.zip")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=BBBC5_IMAGE_URL,
                                             dest_dir=self.data_dir)

        labels = []

        # Read the zip file
        with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()

        file_list = file_list[1:]

        # Get the labels from filenames
        for filename in file_list:
            if filename.split('/')[-1].split('.')[-1] == 'TIF':
                labels.append(int(filename.split('/')[-1].split('_')[2][1:]))

        lbx = np.array(labels, dtype=np.int32)

        loader = dc.data.ImageLoader(sorting=False)
        return loader.create_dataset(inputs=(dataset_file, lbx),
                                     in_memory=False)


def load_bbbc005(
    splitter: Union[dc.splits.Splitter, str, None] = 'index',
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load BBBC005 dataset

    This dataset contains data corresponding to 19,200 samples of synthetically generated
    fluorescent cell population images. These images were simulated for a given cell count
    with a clustering probablity of 25% and a CCD noise variance of 0.0001. Focus blur
    was simulated by applying varying Guassian filters to the images. Each image is of
    size 520x696. Ground truth labels contain cell counts for this dataset. Full details
    about this dataset are present at
    https://data.broadinstitute.org/bbbc/BBBC005/.

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

    Examples
    --------
    Importing necessary modules

    >> import deepchem as dc
    >> import numpy as np

    We will now load the BBBC005 dataset with cell counts as labels.

    >> loader = dc.molnet.load_bbbc005()
    >> tasks, dataset, transformers = loader
    >> train, val, test = dataset

    We now have a dataset with a total of 19,200 samples with cell counts in
    the range of 1-100. The images are of size 520x696. The labels are cell
    counts. We have a train-val-test split of 80:10:10. We can verify this as follows:

    >> train.X.shape
    (15360, 520, 696)
    >> train.y.shape
    (15360,)
    """
    featurizer = dc.feat.UserDefinedFeaturizer([])  # Not actually used
    loader: _MolnetLoader
    loader = _BBBC005_Loader(featurizer, splitter, transformers, BBBC5_TASKS,
                             data_dir, save_dir, **kwargs)

    return loader.load_dataset('bbbc005', reload)
