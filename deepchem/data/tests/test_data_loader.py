"""
Tests for FeaturizedSamples class
"""

import os
import tempfile
import shutil
import deepchem as dc


def test_unlabelled():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "../../data/tests/no_labels.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(tasks=[],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    assert len(dataset.X)


def test_scaffold_test_train_valid_test_split():
    """Test of singletask RF ECFP regression API."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = ["log-solubility"]
    input_file = os.path.join(current_dir,
                              "../../models/tests/assets/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(current_dir, input_file)
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    dataset = loader.create_dataset(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)
    assert len(train_dataset) == 8
    assert len(valid_dataset) == 1
    assert len(test_dataset) == 1


def test_scaffold_test_train_test_split():
    """Test of singletask RF ECFP regression API."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = ["log-solubility"]
    input_file = os.path.join(current_dir,
                              "../../models/tests/assets/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(current_dir, input_file)
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    dataset = loader.create_dataset(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)
    assert len(train_dataset) == 8
    assert len(test_dataset) == 2


def test_random_test_train_valid_test_split():
    """Test of singletask RF ECFP regression API."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = ["log-solubility"]
    input_file = os.path.join(current_dir,
                              "../../models/tests/assets/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)

    input_file = os.path.join(current_dir, input_file)
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    dataset = loader.create_dataset(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset)
    assert len(train_dataset) == 8
    assert len(valid_dataset) == 1
    assert len(test_dataset) == 1


def test_random_test_train_test_split():
    """Test of singletask RF ECFP regression API."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = ["log-solubility"]
    input_file = os.path.join(current_dir,
                              "../../models/tests/assets/example.csv")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)

    dataset = loader.create_dataset(input_file)

    # Splits featurized samples into train/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)
    assert len(train_dataset) == 8
    assert len(test_dataset) == 2


def test_log_solubility_dataset():
    """Test of loading for simple log-solubility dataset."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    input_file = "../../models/tests/assets/example.csv"
    input_file = os.path.join(current_dir, input_file)

    tasks = ["log-solubility"]
    loader = dc.data.CSVLoader(
        tasks=tasks,
        feature_field="smiles",
        featurizer=dc.feat.CircularFingerprint(size=1024))
    dataset = loader.create_dataset(input_file)

    assert len(dataset) == 10


def test_dataset_move():
    """Test that dataset can be moved and reloaded."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = tempfile.mkdtemp()
    data_dir = os.path.join(base_dir, "data")
    moved_data_dir = os.path.join(base_dir, "moved_data")
    dataset_file = os.path.join(current_dir,
                                "../../models/tests/assets/example.csv")

    featurizer = dc.feat.CircularFingerprint(size=1024)
    tasks = ["log-solubility"]
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    featurized_dataset = loader.create_dataset(dataset_file, data_dir)
    n_dataset = len(featurized_dataset)

    # Now perform move
    shutil.move(data_dir, moved_data_dir)

    moved_featurized_dataset = dc.data.DiskDataset(moved_data_dir)

    assert len(moved_featurized_dataset) == n_dataset
