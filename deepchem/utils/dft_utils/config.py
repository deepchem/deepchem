from dataclasses import dataclass

__all__ = ["config"]

@dataclass
class _Config(object):
    # Threshold memory (matrix above this size should not be constructed)
    THRESHOLD_MEMORY: int = 10 * 1024 ** 3  # in B
    # The memory for splitting big tensors into chunks
    CHUNK_MEMORY: int = 16 * 1024 ** 2  # in B

    VERBOSE: int = 0  # verbosity level

config = _Config()
