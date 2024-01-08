import deepchem as dc
from deepchem.data import Dataset
import posixpath
import os
import numpy as np
import ray
from ray.data import Datasink
from ray.data.block import Block, BlockAccessor
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.datasource.filename_provider import FilenameProvider

from typing import Dict, Any, Iterable, Optional, List
from functools import partial

ray.init(num_cpus=4)

# TODO Implement reading and writing data from disk
class RayDataset(Dataset):

    def __init__(
        self,
        dataset,
        x_column: str = 'features',
        y_column: str = 'y',
    ):
        """Initialize this datasink.

        Args:
            dataset: ray.data.Dataset 
        """
        self.ds = dataset
        self.x_column, self.y_column = x_column, y_column

    def iterbatches(self, batch_size: int = 16, epochs=1, deterministic: bool = False, pad_batches: bool = False):
        for batch in self.ds.iter_batches(batch_size=batch_size, batch_format='numpy'):
            x, y = batch[self.x_column], batch[self.y_column]
            w, ids = np.ones(batch_size), np.ones(batch_size)
            yield (x, y, w, ids)


def featurize(row: Dict[str, Any],
              featurizer,
              x='smiles',
              y='logp') -> Dict[str, Any]:
    row['features'] = featurizer(row['smiles'])
    row['y'] = row['logp']
    return row


featurize_batches = partial(featurize, featurizer=dc.feat.CircularFingerprint())

ds = ray.data.read_csv('zinc1k.csv').map_batches(featurize_batches, num_cpus=4)
rds = RayDataset(ds)

for i, batch in enumerate(rds.iterbatches()):
    pass
