"""
Density Function Theory Configuration Utilities.
"""

from dataclasses import dataclass

__all__ = ["config"]


@dataclass
class _Config(object):
    """Contains the configuration for the DFT module

    Examples
    --------
    >>> from deepchem.utils.dft_utils.config import config
    >>> Memory_usage = 1024**4 # Sample Memory usage by some Object/Matrix
    >>> if Memory_usage > config.THRESHOLD_MEMORY :
    ...     print("Overload")
    Overload

    Attributes
    ----------
    THRESHOLD_MEMORY: int (default=10*1024**3)
        Threshold memory (matrix above this size should not be constructed)
    CHUNK_MEMORY: int (default=16*1024**2)
        The memory for splitting big tensors into chunks.
    VERBOSE: int (default=0)
        Allowed Verbosity level (Defines the level of detail)
        Used by Looger for maintaining Logs.

    Usage
    -----
    1. HamiltonCGTO: Usage it for splitting big tensors into chunks.

    """
    THRESHOLD_MEMORY: int = 10 * 1024**3  # in Bytes
    CHUNK_MEMORY: int = 16 * 1024**2  # in Bytes
    VERBOSE: int = 0


config = _Config()
