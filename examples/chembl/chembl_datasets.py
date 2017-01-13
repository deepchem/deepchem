"""
KAGGLE dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import time
import numpy as np
import deepchem as dc
import sys

sys.path.append(".")
from chembl_tasks import chembl_tasks

def remove_missing_entries(dataset):
    """Remove missing entries.

    Some of the datasets have missing entries that sneak in as zero'd out
    feature vectors. Get rid of them.
    """
    for i, (X, y, w, ids) in enumerate(dataset.itershards()):
        available_rows = X.any(axis=1)
        print("Shard %d has %d missing entries."
              % (i, np.count_nonzero(~available_rows)))
        X = X[available_rows]
        y = y[available_rows]
        w = w[available_rows]
        ids = ids[available_rows]
        dataset.set_shard(i, X, y, w, ids)


# Set shard size low to avoid memory problems.
def load_chembl(shard_size=2000, featurizer="ECFP", split='random'):
    """Load KAGGLE datasets. Does not do train/test split"""
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    # Set some global variables up top
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Load dataset
    print("About to load ChEMBL dataset.")
    dataset_path = os.path.join(
        current_dir, "../../datasets/chembl.csv")

    # Featurize KAGGLE dataset
    print("About to featurize ChEMBL dataset.")
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer()

    loader = dc.data.CSVLoader(
        tasks=chembl_tasks, smiles_field="smiles", featurizer=featurizer)

    dataset = loader.featurize(dataset_path, shard_size=shard_size)

    # Initialize transformers
    print("About to transform data")
    transformers = [
        dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]
    for transformer in transformers:
        dataset = transformer.transform(dataset)

    splitters = {'index': dc.splits.IndexSplitter(),
                 'random': dc.splits.RandomSplitter(),
                 'scaffold': dc.splits.ScaffoldSplitter()}
    splitter = splitters[split]
    print("Performing new split.")
    train, valid, test = splitter.train_valid_test_split(dataset)

    return chembl_tasks, (train, valid, test), transformers