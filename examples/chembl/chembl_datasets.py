"""
ChEMBL dataset loader.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time

import deepchem as dc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chembl_tasks import chembl_tasks


# Set shard size low to avoid memory problems.
def load_chembl(shard_size=2000, featurizer="ECFP", set="5thresh", split="random"):
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    # Set some global variables up top
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Load dataset
    print("About to load ChEMBL dataset.")
    if split == "year":
        train_datasets, valid_datasets, test_datasets = [], [], []
        train_files = os.path.join(current_dir,
                                   "year_sets/chembl_%s_ts_train.csv.gz" % set)
        valid_files = os.path.join(current_dir,
                                   "year_sets/chembl_%s_ts_valid.csv.gz" % set)
        test_files = os.path.join(current_dir,
                                  "year_sets/chembl_%s_ts_test.csv.gz" % set)
    else:
        dataset_path = os.path.join(
            current_dir, "../../datasets/chembl_%s.csv.gz" % set)

    # Featurize ChEMBL dataset
    print("About to featurize ChEMBL dataset.")
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer()

    loader = dc.data.CSVLoader(
        tasks=chembl_tasks, smiles_field="smiles", featurizer=featurizer)

    if split == "year":
        print("Featurizing train datasets")
        train_dataset = loader.featurize(
            train_files, shard_size=shard_size)

        print("Featurizing valid datasets")
        valid_dataset = loader.featurize(
            valid_files, shard_size=shard_size)

        print("Featurizing test datasets")
        test_dataset = loader.featurize(
            test_files, shard_size=shard_size)
    else:
        dataset = loader.featurize(dataset_path, shard_size=shard_size)

    # Initialize transformers
    print("About to transform data")
    if split == "year":
        transformers = [
            dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
        for transformer in transformers:
            train = transformer.transform(train_dataset)
            valid = transformer.transform(valid_dataset)
            test = transformer.transform(test_dataset)
    else:
        transformers = [
            dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]
        for transformer in transformers:
            dataset = transformer.transform(dataset)

    splitters = {'index': dc.splits.IndexSplitter(),
                 'random': dc.splits.RandomSplitter(),
                 'scaffold': dc.splits.ScaffoldSplitter()}
    if split in splitters:
        splitter = splitters[split]
        print("Performing new split.")
        train, valid, test = splitter.train_valid_test_split(dataset)


    return chembl_tasks, (train, valid, test), transformers
