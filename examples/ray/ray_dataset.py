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
              y='measured log solubility in mols per litre') -> Dict[str, Any]:
    row['features'] = featurizer(row[x])
    row['y'] = row[y]
    return row


featurize_batches = partial(featurize, featurizer=dc.feat.DummyFeaturizer())

ds = ray.data.read_csv('delaney-processed.csv').map_batches(featurize_batches)
rds = RayDataset(ds)

for i, batch in enumerate(rds.iterbatches()):
    print (i)
