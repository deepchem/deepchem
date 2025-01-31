import os
import pytest
import numpy as np
import joblib
from deepchem.utils.data_utils import save_to_disk

@pytest.fixture
def temp_file(tmp_path):
    """Fixture to create a temporary directory for testing."""
    return str(tmp_path / "test_data.npy")

def test_save_to_new_file(temp_file):
    """Test saving to a new file."""
    data = np.array([1, 2, 3])
    save_to_disk(data, temp_file)
    assert os.path.exists(temp_file)

def test_prevent_overwrite(temp_file):
    """Test that attempting to save to an existing file without `overwrite=True` raises an error."""
    data = np.array([4, 5, 6])
    save_to_disk(data, temp_file)  # Save once

    with pytest.raises(FileExistsError):
        save_to_disk(data, temp_file)  # Try saving again without overwrite=True

def test_allow_overwrite(temp_file):
    """Test that setting `overwrite=True` allows saving to an existing file."""
    data1 = np.array([7, 8, 9])
    data2 = np.array([10, 11, 12])

    save_to_disk(data1, temp_file)  # Save first dataset
    assert os.path.exists(temp_file)

    save_to_disk(data2, temp_file, overwrite=True)  # Overwrite with second dataset
    loaded_data = np.load(temp_file)
    assert np.array_equal(loaded_data, data2)  # Ensure new data was saved

def test_save_joblib(tmp_path):
    """Test saving a dataset as a joblib file."""
    temp_file = str(tmp_path / "test_data.joblib")
    data = {"a": 1, "b": 2}

    save_to_disk(data, temp_file)
    assert os.path.exists(temp_file)

    loaded_data = joblib.load(temp_file)
    assert loaded_data == data  # Ensure correct data is saved

def test_invalid_extension(tmp_path):
    """Test that an unsupported file extension raises a ValueError."""
    temp_file = str(tmp_path / "test_data.txt")
    data = {"key": "value"}

    with pytest.raises(ValueError):
        save_to_disk(data, temp_file)
