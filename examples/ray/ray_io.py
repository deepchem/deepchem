import deepchem as dc
import os
import ray

from typing import Dict, Any, Iterable, Optional, List
from functools import partial


ray.init(num_cpus=4)

def featurize(row: Dict[str, Any],
              featurizer,
              x='smiles',
              y='logp') -> Dict[str, Any]:
    return row


featurize_batches = partial(featurize, featurizer=dc.feat.CircularFingerprint())

# Featurizing a dataset
ds = ray.data.read_csv('zinc1k.csv').map_batches(featurize_batches, num_cpus=4)

# Writing a dataset to disk
ds.write_parquet('data-dir')

# Reading a dataset from disk
ds = ray.data.read_parquet('data-dir')

for i, batch in enumerate(ds.iter_batches()):
    print (i)

ray.shutdown()
