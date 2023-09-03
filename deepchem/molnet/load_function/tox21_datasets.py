"""
Tox21 dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]


class _Tox21Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "tox21.csv.gz")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=TOX21_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_tox21(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    tasks: List[str] = TOX21_TASKS,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load Tox21 dataset

    The "Toxicology in the 21st Century" (Tox21) initiative created a public
    database measuring toxicity of compounds, which has been used in the 2014
    Tox21 Data Challenge. This dataset contains qualitative toxicity measurements
    for 8k compounds on 12 different targets, including nuclear receptors and
    stress response pathways.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles" - SMILES representation of the molecular structure
    - "NR-XXX" - Nuclear receptor signaling bioassays results
    - "SR-XXX" - Stress response bioassays results

    please refer to https://tripod.nih.gov/tox21/challenge/data.jsp for details.

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
    tasks: List[str], (optional)
        Specify the set of tasks to load. If no task is specified, then it loads
    the default set of tasks which are NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER,
    NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53.

    References
    ----------
    .. [1] Tox21 Challenge. https://tripod.nih.gov/tox21/challenge/
    """
    loader = _Tox21Loader(featurizer, splitter, transformers, tasks, data_dir,
                          save_dir, **kwargs)
    return loader.load_dataset('tox21', reload)
