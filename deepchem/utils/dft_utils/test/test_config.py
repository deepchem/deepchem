"""
Test for DFT Config Utilities
"""

from deepchem.utils.dft_utils.config import config


def test_config():
    conf = config
    assert conf.THRESHOLD_MEMORY == 10 * 1024**3
    assert conf.CHUNK_MEMORY == 16 * 1024**2
    assert conf.VERBOSE == 0
