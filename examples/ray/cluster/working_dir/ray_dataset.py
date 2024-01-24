import deepchem as dc
import os
import ray

from typing import Dict, Any
from functools import partial


def featurize(row, featurizer, x='smiles', y='logp') -> Dict[str, Any]:
    row['features'] = featurizer(row[x])
    row['y'] = row[y]
    return row


featurize_batches = partial(featurize, featurizer=dc.feat.DummyFeaturizer())

ds = ray.data.read_csv('zinc1k.csv').map_batches(featurize_batches)

count = ds.count()
print('total elements is ', count)

print('current working dir is ', os.getcwd())
# ds.write_parquet('s3://chemberta3/ray-data/zinc1k') -- this throws aws authentication error https://stackoverflow.com/questions/33600192/aws-s3-cli-anonymous-users-cannot-initiate-multipart-uploads
# write to a data dir
ds.write_parquet('local:///home/ubuntu/data')
# Upload data to s3 bucket from home directory
print('wrote dataset')
