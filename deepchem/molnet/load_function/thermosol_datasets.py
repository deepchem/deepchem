"""
Thermodynamic Solubility Dataset Loader
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

THERMOSOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/thermosol.csv"
THERMOSOL_TASKS = ["target"]  # Task is solubility in pH 7.4 buffer


class _ThermosolLoader(_MolnetLoader):
    def __init__(self, featurizer, *args, **kwargs):
        super(_ThermosolLoader, self).__init__(*args, **kwargs)
        self.featurizer = featurizer
            

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "thermosol.csv")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=THERMOSOL_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smile",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_thermosol(
    featurizer: Union["dc.feat.Featurizer", str] = 'ECFP',
    splitter: Union["dc.splits.Splitter", str, None] = 'scaffold',
    transformers: List[Union["TransformerGenerator", str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple["Dataset", ...], List["dc.trans.Transformer"]]:
    """Loads the thermodynamic solubility datasets.

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
    """
    loader = _ThermosolLoader(featurizer, splitter, transformers,
                              THERMOSOL_TASKS, data_dir, save_dir, **kwargs)
    return loader.load_dataset('thermosol', reload)
