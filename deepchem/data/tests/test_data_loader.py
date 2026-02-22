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


def test_csv_loader_respects_shard_size():
    """Test that CSVLoader properly respects the shard_size parameter."""
    # Create a temporary CSV file with 20 molecules
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False) as f:
        f.write("smiles,label\n")
        for i in range(20):
            f.write(f"C{'C' * i},{i}\n")
        csv_file = f.name

    try:
        with tempfile.TemporaryDirectory() as data_dir:
            featurizer = dc.feat.CircularFingerprint(size=1024)
            loader = dc.data.CSVLoader(tasks=["label"],
                                       feature_field="smiles",
                                       featurizer=featurizer)

            # Load with shard_size=5 (should create 4 shards: 5+5+5+5)
            dataset = loader.create_dataset(csv_file,
                                            data_dir=data_dir,
                                            shard_size=5)

            # Verify total length
            assert len(dataset) == 20, \
                f"Expected 20 samples, got {len(dataset)}"

            # Verify number of shards created
            shard_files = [
                f for f in os.listdir(data_dir) if f.startswith("shard")
            ]
            assert len(shard_files) == 4, \
                f"Expected 4 shards, got {len(shard_files)}"
    finally:
        os.unlink(csv_file)


def test_fasta_loader_respects_shard_size():
    """Test that FASTALoader properly respects the shard_size parameter.

    This test addresses issue #4741 - ensuring that the shard_size parameter
    actually works for FASTALoader instead of being ignored.
    """
    # Create a test FASTA file with 10 sequences
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta',
                                     delete=False) as f:
        for i in range(10):
            f.write(f">sequence_{i}\n")
            f.write("ATCGATCGATCGATCG\n")
        fasta_file = f.name

    try:
        # Test 1: Loading without shard_size (should be 1 shard)
        with tempfile.TemporaryDirectory() as data_dir1:
            loader = dc.data.FASTALoader(featurizer=None,
                                         auto_add_annotations=False,
                                         legacy=False)
            dataset1 = loader.create_dataset(fasta_file, data_dir=data_dir1)

            assert len(dataset1) == 10, \
                f"Expected 10 sequences, got {len(dataset1)}"

            shard_files = [
                f for f in os.listdir(data_dir1) if f.startswith("shard")
            ]
            assert len(shard_files) == 1, \
                f"Expected 1 shard without shard_size, got {len(shard_files)}"

        # Test 2: Loading with shard_size=3 (should create 4 shards: 3+3+3+1)
        with tempfile.TemporaryDirectory() as data_dir2:
            loader = dc.data.FASTALoader(featurizer=None,
                                         auto_add_annotations=False,
                                         legacy=False)
            dataset2 = loader.create_dataset(fasta_file,
                                             data_dir=data_dir2,
                                             shard_size=3)

            assert len(dataset2) == 10, \
                f"Expected 10 sequences, got {len(dataset2)}"

            shard_files = [
                f for f in os.listdir(data_dir2) if f.startswith("shard")
            ]
            assert len(shard_files) == 4, \
                f"Expected 4 shards with shard_size=3, got {len(shard_files)}"

        # Test 3: Verify both datasets contain the same data
        assert len(dataset1) == len(dataset2), \
            "Datasets with different shard_size should have same total length"

    finally:
        os.unlink(fasta_file)


def test_fasta_loader_shard_size_edge_cases():
    """Test edge cases for FASTALoader shard_size parameter."""
    # Test with empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta',
                                     delete=False) as f:
        empty_file = f.name

    try:
        with tempfile.TemporaryDirectory() as data_dir:
            loader = dc.data.FASTALoader(featurizer=None,
                                         auto_add_annotations=False,
                                         legacy=False)
            dataset = loader.create_dataset(empty_file,
                                            data_dir=data_dir,
                                            shard_size=5)
            assert len(dataset) == 0, \
                f"Expected 0 sequences from empty file, got {len(dataset)}"
    finally:
        os.unlink(empty_file)

    # Test with single sequence
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta',
                                     delete=False) as f:
        f.write(">single\n")
        f.write("ATCG\n")
        single_file = f.name

    try:
        with tempfile.TemporaryDirectory() as data_dir:
            loader = dc.data.FASTALoader(featurizer=None,
                                         auto_add_annotations=False,
                                         legacy=False)
            dataset = loader.create_dataset(single_file,
                                            data_dir=data_dir,
                                            shard_size=10)
            assert len(dataset) == 1, \
                f"Expected 1 sequence, got {len(dataset)}"
    finally:
        os.unlink(single_file)

    # Test with exact shard_size division (10 sequences, shard_size=5)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta',
                                     delete=False) as f:
        for i in range(10):
            f.write(f">seq_{i}\n")
            f.write("ATCG\n")
        exact_file = f.name

    try:
        with tempfile.TemporaryDirectory() as data_dir:
            loader = dc.data.FASTALoader(featurizer=None,
                                         auto_add_annotations=False,
                                         legacy=False)
            dataset = loader.create_dataset(exact_file,
                                            data_dir=data_dir,
                                            shard_size=5)
            assert len(dataset) == 10, \
                f"Expected 10 sequences, got {len(dataset)}"

            shard_files = [
                f for f in os.listdir(data_dir) if f.startswith("shard")
            ]
            assert len(shard_files) == 2, \
                f"Expected 2 shards (5+5), got {len(shard_files)}"
    finally:
        os.unlink(exact_file)
