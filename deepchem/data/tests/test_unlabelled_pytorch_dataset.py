"""Regression tests for iterating unlabeled disk data without changing storage."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import deepchem as dc

ITERATORS = ["native", pytest.param("torch", marks=pytest.mark.torch)]


def _dataset(path, representations=("none", "none"), sizes=(3, 2)):
    """Create distinguishable rows with explicit absence representations."""
    count = sum(sizes)
    ids = np.arange(count)
    X = np.column_stack((ids + 0.25, ids + 10.5)).astype(np.float32)

    def shards():
        offset = 0
        for representation, size in zip(representations, sizes):
            if representation == "none":
                y, w = None, None
            elif representation == "empty":
                y = np.array([], dtype=np.int16)
                w = np.array([], dtype=np.float32)
            elif representation == "columns":
                y = np.empty((size, 0), dtype=np.int16)
                w = np.empty((size, 0), dtype=np.float32)
            else:
                y = (ids[offset:offset + size] * 2 + 0.5).astype(
                    np.float32).reshape(size, 1)
                w = (ids[offset:offset + size] + 1).astype(np.float32).reshape(
                    size, 1)
            yield X[offset:offset + size], y, w, ids[offset:offset + size]
            offset += size

    tasks = ["target"] if representations[0] == "labeled" else []
    dataset = dc.data.DiskDataset.create_dataset(shards(),
                                                 data_dir=str(path),
                                                 tasks=tasks)
    return dataset, X, ids


def _iterate(dataset, iterator, batch_size, epochs=2):
    if iterator == "torch":
        pytest.importorskip("torch")
        return dataset.make_pytorch_dataset(batch_size=batch_size,
                                            epochs=epochs,
                                            deterministic=True)
    return dataset.iterbatches(batch_size=batch_size,
                               epochs=epochs,
                               deterministic=True,
                               pad_batches=False)


def _joined(items, field, iterator, batch_size):
    values = [np.asarray(item[field]) for item in items]
    if iterator == "torch" and batch_size is None:
        return np.stack(values)
    return np.concatenate(values)


def _file_contents(dataset):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(dataset.data_dir).iterdir()
        if path.is_file()
    }


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("representations",
                         [("none", "none"), ("empty", "empty"),
                          ("none", "empty"), ("empty", "none")])
def test_unlabelled_samples_and_batches(tmp_path, iterator, batch_size,
                                        representations):
    """Both missing-field forms survive shard carry and repeated epochs."""
    dataset, X, ids = _dataset(tmp_path / "data", representations)
    items = list(_iterate(dataset, iterator, batch_size))
    assert all(item[1] is None and item[2] is None for item in items)
    np.testing.assert_array_equal(_joined(items, 0, iterator, batch_size),
                                  np.concatenate([X, X]))
    np.testing.assert_array_equal(_joined(items, 3, iterator, batch_size),
                                  np.concatenate([ids, ids]))


@pytest.mark.parametrize("iterator,batch_size",
                         [("native", 2),
                          pytest.param("torch", None, marks=pytest.mark.torch)])
@pytest.mark.parametrize("field,representation", [(1, "none"), (2, "none"),
                                                  (1, "empty"), (2, "empty")])
def test_independent_missing_fields(tmp_path, iterator, batch_size, field,
                                    representation):
    """Absence of either field must not discard the other field's data."""
    ids = np.arange(3)
    X = np.column_stack((ids + 0.25, ids + 10.5)).astype(np.float32)
    y = (ids * 2 + 0.5).astype(np.float32).reshape(-1, 1)
    w = (ids + 1).astype(np.float32).reshape(-1, 1)
    shard = [X, y, w, ids]
    shard[field] = None if representation == "none" else np.array([])
    dataset = dc.data.DiskDataset.create_dataset(
        [tuple(shard)],
        tasks=[] if field == 1 else ["target"],
        data_dir=str(tmp_path / "data"))
    items = list(_iterate(dataset, iterator, batch_size))
    assert all(item[field] is None for item in items)
    other = 3 - field
    np.testing.assert_array_equal(_joined(items, other, iterator, batch_size),
                                  np.concatenate([shard[other], shard[other]]))
    np.testing.assert_array_equal(_joined(items, 0, iterator, batch_size),
                                  np.concatenate([X, X]))
    np.testing.assert_array_equal(_joined(items, 3, iterator, batch_size),
                                  np.concatenate([ids, ids]))


@pytest.mark.parametrize("representation", ["none", "empty"])
def test_unlabelled_padding(tmp_path, representation):
    """Padding keeps fields absent while repeating only the final real row."""
    dataset, X, ids = _dataset(tmp_path / "data",
                               (representation, representation))
    items = list(
        dataset.iterbatches(batch_size=2,
                            epochs=2,
                            deterministic=True,
                            pad_batches=True))
    assert all(item[1] is None and item[2] is None for item in items)
    padded_X = np.concatenate([X, X[-1:]])
    padded_ids = np.concatenate([ids, ids[-1:]])
    np.testing.assert_array_equal(_joined(items, 0, "native", 2),
                                  np.concatenate([padded_X, padded_X]))
    np.testing.assert_array_equal(_joined(items, 3, "native", 2),
                                  np.concatenate([padded_ids, padded_ids]))


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("representation", ["none", "empty"])
def test_unlabelled_default_random_order(tmp_path, iterator, batch_size,
                                         representation):
    """The default shuffled iteration preserves rows and IDs across epochs."""
    dataset, X, ids = _dataset(tmp_path / "data",
                               (representation, representation))
    if iterator == "torch":
        pytest.importorskip("torch")
        items = list(
            dataset.make_pytorch_dataset(batch_size=batch_size, epochs=2))
    else:
        items = list(
            dataset.iterbatches(batch_size=batch_size,
                                epochs=2,
                                pad_batches=False))
    assert all(item[1] is None and item[2] is None for item in items)
    observed_ids = _joined(items, 3, iterator, batch_size)
    np.testing.assert_array_equal(np.sort(observed_ids),
                                  np.sort(np.concatenate([ids, ids])))
    np.testing.assert_array_equal(_joined(items, 0, iterator, batch_size),
                                  X[observed_ids.astype(int)])


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_csv_random_split_unlabelled(tmp_path, iterator, batch_size):
    """Exercise the original loader/split path without downloading USPTO."""
    reactions = [
        "C>>C", "CC>>CC", "CCC>>CCC", "CCCC>>CCCC", "CO>>C=O", "CCO>>CC=O",
        "CN>>C#N", "CCN>>CC#N"
    ]
    csv = tmp_path / "reactions.csv"
    pd.DataFrame({
        "id": ["r" + str(i) for i in range(8)],
        "reactions": reactions
    }).to_csv(csv, index=False)
    loader = dc.data.CSVLoader(tasks=[],
                               feature_field="reactions",
                               id_field="id",
                               featurizer=dc.feat.DummyFeaturizer())
    loaded = loader.create_dataset(str(csv),
                                   data_dir=str(tmp_path / "loaded"),
                                   shard_size=2)
    dataset, _, _ = dc.splits.RandomSplitter().train_valid_test_split(
        loaded,
        frac_train=0.5,
        frac_valid=0.25,
        frac_test=0.25,
        seed=0,
        train_dir=str(tmp_path / "train"),
        valid_dir=str(tmp_path / "valid"),
        test_dir=str(tmp_path / "test"))
    ids = dataset.ids.copy()
    expected = np.array([reactions[int(identifier[1:])] for identifier in ids])
    assert all(
        dataset.get_shard(i)[1].shape == (
            0,) and dataset.get_shard(i)[2].shape == (0,)
        for i in range(dataset.get_number_shards()))
    items = list(_iterate(dataset, iterator, batch_size))
    assert all(item[1] is None and item[2] is None for item in items)
    np.testing.assert_array_equal(_joined(items, 0, iterator, batch_size),
                                  np.concatenate([expected, expected]))
    np.testing.assert_array_equal(_joined(items, 3, iterator, batch_size),
                                  np.concatenate([ids, ids]))


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("batch_size", [None, 2])
def test_legacy_absence_preserves_cache_getters_and_files(
        tmp_path, warm_cache, batch_size):
    """Iteration must not rewrite legacy raw arrays, getters or metadata."""
    dataset, X, ids = _dataset(tmp_path / "data", ("empty", "empty"))
    before = _file_contents(dataset)
    metadata = dataset.metadata_df.copy(deep=True)
    cold_y = dataset.y.copy()
    cold_w = dataset.w.copy()
    assert cold_y.shape == cold_w.shape == (0,)
    assert dataset._cached_shards is None
    cached = None
    if warm_cache:
        cached = [
            dataset.get_shard(i) for i in range(dataset.get_number_shards())
        ]
    items = list(_iterate(dataset, "native", batch_size))
    assert all(item[1] is None and item[2] is None for item in items)
    np.testing.assert_array_equal(_joined(items, 0, "native", batch_size),
                                  np.concatenate([X, X]))
    np.testing.assert_array_equal(_joined(items, 3, "native", batch_size),
                                  np.concatenate([ids, ids]))
    for i in range(dataset.get_number_shards()):
        raw = dataset.get_shard(i)
        assert raw[1].shape == raw[2].shape == (0,)
        assert raw[1].dtype == np.int16
        assert raw[2].dtype == np.float32
        assert dataset.get_shard_y(i) is raw[1]
        assert dataset.get_shard_w(i) is raw[2]
        if cached is not None:
            assert raw[1] is cached[i][1]
            assert raw[2] is cached[i][2]
    np.testing.assert_array_equal(dataset.y, cold_y)
    np.testing.assert_array_equal(dataset.w, cold_w)
    assert dataset.y.dtype == cold_y.dtype
    assert dataset.w.dtype == cold_w.dtype
    assert dataset.metadata_df.equals(metadata)
    assert _file_contents(dataset) == before


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_zero_task_columns_remain_arrays(tmp_path, iterator, batch_size):
    """An array with rows and zero columns is not a missing-field sentinel."""
    dataset, X, ids = _dataset(tmp_path / "data", ("columns", "columns"))
    items = list(_iterate(dataset, iterator, batch_size))
    expected_ndim = 1 if iterator == "torch" and batch_size is None else 2
    for batch_X, y, w, _ in items:
        assert y is not None and w is not None
        assert y.ndim == w.ndim == expected_ndim
        assert y.shape[-1] == w.shape[-1] == 0
        if expected_ndim == 2:
            assert y.shape[0] == w.shape[0] == batch_X.shape[0]
        assert y.dtype == np.int16
        assert w.dtype == np.float32
    np.testing.assert_array_equal(_joined(items, 0, iterator, batch_size),
                                  np.concatenate([X, X]))
    np.testing.assert_array_equal(_joined(items, 3, iterator, batch_size),
                                  np.concatenate([ids, ids]))


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("representation",
                         ["none", "empty", "columns", "labeled"])
def test_zero_row_shard_stays_empty(tmp_path, iterator, batch_size,
                                    representation):
    """Standalone zero-row shards yield no samples and retain raw shapes."""
    dataset, _, _ = _dataset(tmp_path / "data", (representation,), (0,))
    raw = dataset.get_shard(0)
    before = _file_contents(dataset)
    assert list(_iterate(dataset, iterator, batch_size)) == []
    after = dataset.get_shard(0)
    assert after[1] is raw[1] and after[2] is raw[2]
    assert _file_contents(dataset) == before


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("representation", ["none", "empty"])
@pytest.mark.parametrize("sizes", [(0, 3, 2), (3, 0, 2), (3, 0)])
def test_unlabelled_zero_row_between_batches(tmp_path, iterator, batch_size,
                                             representation, sizes):
    """Empty legacy shards preserve pending rows and the raw cached fields."""
    representations = tuple(
        "empty" if size == 0 else representation for size in sizes)
    dataset, X, ids = _dataset(tmp_path / "data", representations, sizes)
    raw = [dataset.get_shard(i) for i in range(len(sizes))]
    before = _file_contents(dataset)
    items = list(_iterate(dataset, iterator, batch_size))
    assert all(item[1] is None and item[2] is None for item in items)
    np.testing.assert_array_equal(_joined(items, 0, iterator, batch_size),
                                  np.concatenate([X, X]))
    np.testing.assert_array_equal(_joined(items, 3, iterator, batch_size),
                                  np.concatenate([ids, ids]))
    for i in range(len(sizes)):
        after = dataset.get_shard(i)
        assert after[1] is raw[i][1] and after[2] is raw[i][2]
    assert _file_contents(dataset) == before


@pytest.mark.parametrize("representation", ["none", "empty"])
def test_unlabelled_trailing_zero_row_padding(tmp_path, representation):
    """A pending final row is flushed and padded through a trailing empty shard."""
    dataset, X, ids = _dataset(tmp_path / "data", (representation, "empty"),
                               (3, 0))
    items = list(
        dataset.iterbatches(batch_size=2,
                            epochs=2,
                            deterministic=True,
                            pad_batches=True))
    assert all(item[1] is None and item[2] is None for item in items)
    padded_X = np.concatenate([X, X[-1:]])
    padded_ids = np.concatenate([ids, ids[-1:]])
    np.testing.assert_array_equal(_joined(items, 0, "native", 2),
                                  np.concatenate([padded_X, padded_X]))
    np.testing.assert_array_equal(_joined(items, 3, "native", 2),
                                  np.concatenate([padded_ids, padded_ids]))


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("missing_field", [1, 2])
def test_zero_row_carry_fields_independent(tmp_path, iterator, missing_field):
    """A zero-row sentinel does not discard the carried other field's values."""
    ids = np.arange(5)
    X = np.column_stack((ids + 0.25, ids + 10.5)).astype(np.float32)
    other_values = (ids * 3 + 0.75).astype(np.float32).reshape(-1, 1)
    shards = []
    for start, end in [(0, 3), (3, 3), (3, 5)]:
        shard = [
            X[start:end], other_values[start:end], other_values[start:end],
            ids[start:end]
        ]
        shard[missing_field] = np.array([], dtype=np.float32)
        shards.append(tuple(shard))
    dataset = dc.data.DiskDataset.create_dataset(shards,
                                                 tasks=["target"],
                                                 data_dir=str(tmp_path /
                                                              "data"))
    items = list(_iterate(dataset, iterator, 2))
    assert all(item[missing_field] is None for item in items)
    np.testing.assert_array_equal(
        _joined(items, 3 - missing_field, iterator, 2),
        np.concatenate([other_values, other_values]))
    np.testing.assert_array_equal(_joined(items, 0, iterator, 2),
                                  np.concatenate([X, X]))
    np.testing.assert_array_equal(_joined(items, 3, iterator, 2),
                                  np.concatenate([ids, ids]))


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_labelled_alignment_unchanged(tmp_path, iterator, batch_size):
    """Nonconstant labels and weights remain aligned with features and IDs."""
    dataset, X, ids = _dataset(tmp_path / "data", ("labeled", "labeled"))
    items = list(_iterate(dataset, iterator, batch_size))
    repeated_ids = np.concatenate([ids, ids])
    np.testing.assert_array_equal(_joined(items, 0, iterator, batch_size),
                                  np.concatenate([X, X]))
    np.testing.assert_array_equal(_joined(items, 3, iterator, batch_size),
                                  repeated_ids)
    np.testing.assert_array_equal(_joined(items, 1, iterator, batch_size),
                                  (repeated_ids * 2 + 0.5).astype(
                                      np.float32).reshape(-1, 1))
    np.testing.assert_array_equal(_joined(items, 2, iterator, batch_size),
                                  (repeated_ids + 1).astype(np.float32).reshape(
                                      -1, 1))


@pytest.mark.parametrize("iterator", ITERATORS)
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("field", [1, 2])
def test_nonempty_misaligned_fields_still_fail(tmp_path, iterator, batch_size,
                                               field):
    """Compatibility must not swallow a genuinely short nonempty field."""
    shard = [np.ones((5, 2)), np.ones((5, 1)), np.ones((5, 1)), np.arange(5)]
    shard[field] = np.ones((2, 1))
    dataset = dc.data.DiskDataset.create_dataset([tuple(shard)],
                                                 tasks=["target"],
                                                 data_dir=str(tmp_path /
                                                              "data"))
    with pytest.raises(IndexError):
        list(_iterate(dataset, iterator, batch_size))
