import os
import deepchem as dc


def test_sdf_load():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=16)
    loader = dc.data.SDFLoader(["LogP(RRCK)"],
                               featurizer=featurizer,
                               sanitize=True)
    dataset = loader.create_dataset(
        os.path.join(current_dir, "membrane_permeability.sdf"))
    assert len(dataset) == 2


def test_singleton_sdf_load():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=16)
    loader = dc.data.SDFLoader(["LogP(RRCK)"],
                               featurizer=featurizer,
                               sanitize=True)
    dataset = loader.create_dataset(os.path.join(current_dir, "singleton.sdf"))
    assert len(dataset) == 1


def test_singleton_sdf_zip_load():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=16)
    loader = dc.data.SDFLoader(["LogP(RRCK)"],
                               featurizer=featurizer,
                               sanitize=True)
    dataset = loader.create_dataset(os.path.join(current_dir, "singleton.zip"))
    assert len(dataset) == 1


def test_sharded_sdf_load():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=16)
    loader = dc.data.SDFLoader(["LogP(RRCK)"],
                               featurizer=featurizer,
                               sanitize=True)
    dataset = loader.create_dataset(os.path.join(current_dir,
                                                 "membrane_permeability.sdf"),
                                    shard_size=1)
    assert dataset.get_number_shards() == 2
    assert len(dataset) == 2


def test_sharded_multi_file_sdf_load():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=16)
    loader = dc.data.SDFLoader(["LogP(RRCK)"],
                               featurizer=featurizer,
                               sanitize=True)
    input_files = [
        os.path.join(current_dir, "membrane_permeability.sdf"),
        os.path.join(current_dir, "singleton.sdf")
    ]
    dataset = loader.create_dataset(input_files, shard_size=1)
    assert dataset.get_number_shards() == 3
    assert len(dataset) == 3


def test_sharded_multi_file_sdf_zip_load():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=16)
    loader = dc.data.SDFLoader(["LogP(RRCK)"],
                               featurizer=featurizer,
                               sanitize=True)
    dataset = loader.create_dataset(os.path.join(current_dir,
                                                 "multiple_sdf.zip"),
                                    shard_size=1)
    assert dataset.get_number_shards() == 3
    assert len(dataset) == 3


def test_sdf_load_with_csv():
    """Test a case where SDF labels are in associated csv file"""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    featurizer = dc.feat.CircularFingerprint(size=16)
    loader = dc.data.SDFLoader(["atomization_energy"],
                               featurizer=featurizer,
                               sanitize=True)
    dataset = loader.create_dataset(os.path.join(current_dir, "water.sdf"),
                                    shard_size=1)
    assert len(dataset) == 10
    assert dataset.get_number_shards() == 10
    assert dataset.get_task_names() == ["atomization_energy"]
