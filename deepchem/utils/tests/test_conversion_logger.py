import os
import pytest
from deepchem.utils.convert.conversion_logger import ConversionErrorLogger

def test_conversion_error_logger(tmp_path):
    test_file = tmp_path / "errors.txt"
    logger = ConversionErrorLogger(test_file)

    logger.log("C1CC1", "some_function", "Ring closure error")
    logger.log("C1=CC=CN=C1", "another_function", "Unknown atom")

    # Flush manually (optional, but good for testing)
    logger.flush()

    # Check file contents
    with open(test_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert "C1CC1" in lines[0]
        assert "some_function" in lines[0]
        assert "Ring closure error" in lines[0]
