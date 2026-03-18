"""
NCI dataset loader.
Original Author - Bharath Ramsundar
Author - Aneesh Pappu
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

NCI_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/nci_unique.csv"
NCI_TASKS = [
    'CCRF-CEM', 'HL-60(TB)', 'K-562', 'MOLT-4', 'RPMI-8226', 'SR', 'A549/ATCC',
    'EKVX', 'HOP-62', 'HOP-92', 'NCI-H226', 'NCI-H23', 'NCI-H322M', 'NCI-H460',
    'NCI-H522', 'COLO 205', 'HCC-2998', 'HCT-116', 'HCT-15', 'HT29', 'KM12',
    'SW-620', 'SF-268', 'SF-295', 'SF-539', 'SNB-19', 'SNB-75', 'U251',
    'LOX IMVI', 'MALME-3M', 'M14', 'MDA-MB-435', 'SK-MEL-2', 'SK-MEL-28',
    'SK-MEL-5', 'UACC-257', 'UACC-62', 'IGR-OV1', 'OVCAR-3', 'OVCAR-4',
    'OVCAR-5', 'OVCAR-8', 'NCI/ADR-RES', 'SK-OV-3', '786-0', 'A498', 'ACHN',
    'CAKI-1', 'RXF 393', 'SN12C', 'TK-10', 'UO-31', 'PC-3', 'DU-145', 'MCF7',
    'MDA-MB-231/ATCC', 'MDA-MB-468', 'HS 578T', 'BT-549', 'T-47D'
]


class _NCILoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "nci_unique.csv")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=NCI_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_nci(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load NCI dataset.

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
    loader = _NCILoader(featurizer, splitter, transformers, NCI_TASKS, data_dir,
                        save_dir, **kwargs)
    return loader.load_dataset('nci', reload)
