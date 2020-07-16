import numpy as np
from typing import List, Optional
from deepchem.data import DiskDataset
from deepchem.data.data_loader.base_loader import DataLoader
from deepchem.utils.genomics import encode_fasta_sequence


class FASTALoader(DataLoader):
  """Handles loading of FASTA files.

  FASTA files are commonly used to hold sequence data. This
  class provides convenience files to lead FASTA data and
  one-hot encode the genomic sequences for use in downstream
  learning tasks.
  """

  def __init__(self):
    """Initialize loader."""
    pass

  def create_dataset(self,
                     input_files: List[str],
                     data_dir: Optional[str] = None,
                     shard_size: Optional[int] = None) -> DiskDataset:
    """Creates a `Dataset` from input FASTA files.

    At present, FASTA support is limited and only allows for one-hot
    featurization, and doesn't allow for sharding.

    Parameters
    ----------
    input_files: list[str]
      List of fasta files.
    data_dir: str, optional
      Name of directory where featurized data is stored.
    shard_size: int, optional
      For now, this argument is ignored and each FASTA file gets its
      own shard. 

    Returns
    -------
    A `Dataset` object containing a featurized representation of data
    from `input_files`.
    """
    if not isinstance(input_files, list):
      input_files = [input_files]

    def shard_generator():
      for input_file in input_files:
        X = encode_fasta_sequence(input_file)
        ids = np.ones(len(X))
        # (X, y, w, ids)
        yield X, None, None, ids

    return DiskDataset.create_dataset(shard_generator(), data_dir)
