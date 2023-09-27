from dataclasses import dataclass

__all__ = ["config"]


@dataclass
class _Config(object):
    """Contains the configuration for the DFT module

    Attributes
    ----------
    THRESHOLD_MEMORY: int
        Threshold memory (matrix above this size should not be constructed)
    CHUNK_MEMORY: int
        The memory for splitting big tensors into chunks
    VERBOSE: int
        Verbosity level

    """
    THRESHOLD_MEMORY: int = 10 * 1024**3  # in Bytes
    CHUNK_MEMORY: int = 16 * 1024**2  # in Bytes
    VERBOSE: int = 0


config = _Config()
