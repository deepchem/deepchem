"""
freesolv dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

FREESOLV_URL = 'https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz'
FREESOLV_TASKS = ['y']

class _FreesolvLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, 'freesolv.csv.gz')
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url = FREESOLV_URL, dest_dir = self.data_dir)
            loader = dc.data.CSVLoader(tasks = self.tasks, feature_field = 'smiles', featurizer = self.featurizer)
            return loader.create_dataset(dataset_file)
    
def load_freesolv(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.MATFeaturizer(),
    splitter: Union[dc.splits.Splitter, str, None] = None,
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:

    loader = _FreesolvLoader(featurizer, splitter, transformers, FREESOLV_TASKS, data_dir, save_dir, **kwargs)
    return loader.load_dataset('freesolv', reload)