"""
Simple utils to save and load from disk.
"""
import joblib
import gzip
import json
import pickle
import pandas as pd
import numpy as np
import os
import deepchem
import warnings
import logging
from typing import List, Optional, Iterator, Any

from deepchem.utils.genomics_utils import encode_bio_sequence as encode_sequence, \
  seq_one_hot_encode as seq_one_hotencode

logger = logging.getLogger(__name__)


def save_to_disk(dataset, filename, compress=3):
  """Save a dataset to file."""
  if filename.endswith('.joblib'):
    joblib.dump(dataset, filename, compress=compress)
  elif filename.endswith('.npy'):
    np.save(filename, dataset)
  else:
    raise ValueError("Filename with unsupported extension: %s" % filename)


def get_input_type(input_file):
  """Get type of input file. Must be csv/pkl.gz/sdf file."""
  filename, file_extension = os.path.splitext(input_file)
  # If gzipped, need to compute extension again
  if file_extension == ".gz":
    filename, file_extension = os.path.splitext(filename)
  if file_extension == ".csv":
    return "csv"
  elif file_extension == ".pkl":
    return "pandas-pickle"
  elif file_extension == ".joblib":
    return "pandas-joblib"
  elif file_extension == ".sdf":
    return "sdf"
  else:
    raise ValueError("Unrecognized extension %s" % file_extension)


def load_data(input_files: List[str],
              shard_size: Optional[int] = None) -> Iterator[Any]:
  """Loads data from disk.

  For CSV files, supports sharded loading for large files.

  Parameters
  ----------
  input_files: list
    List of filenames.
  shard_size: int, optional (default None)
    Size of shard to yield

  Returns
  -------
  Iterator which iterates over provided files.
  """
  if not len(input_files):
    return
  input_type = get_input_type(input_files[0])
  if input_type == "sdf":
    if shard_size is not None:
      logger.info("Ignoring shard_size for sdf input.")
    for value in load_sdf_files(input_files):
      yield value
  elif input_type == "csv":
    for value in load_csv_files(input_files, shard_size):
      yield value
  elif input_type == "pandas-pickle":
    for input_file in input_files:
      yield load_pickle_from_disk(input_file)


def load_image_files(image_files: List[str]) -> np.ndarray:
  """Loads a set of images from disk.

  Parameters
  ----------
  image_files: List[str]
    List of image filenames to load.

  Returns
  -------
  np.ndarray
    A numpy array that contains loaded images. The shape is, `(N,...)`.

  Notes
  -----
  This method requires Pillow to be installed.
  """
  try:
    from PIL import Image
  except ModuleNotFoundError:
    raise ValueError("This function requires Pillow to be installed.")

  images = []
  for image_file in image_files:
    _, extension = os.path.splitext(image_file)
    extension = extension.lower()
    if extension == ".png":
      image = np.array(Image.open(image_file))
      images.append(image)
    elif extension == ".tif":
      im = Image.open(image_file)
      imarray = np.array(im)
      images.append(imarray)
    else:
      raise ValueError("Unsupported image filetype for %s" % image_file)
  return np.array(images)


def load_sdf_files(input_files: List[str],
                   clean_mols: bool = True,
                   tasks: List[str] = [],
                   shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
  """Load SDF file into dataframe.

  Parameters
  ----------
  input_files: list[str]
    List of filenames
  clean_mols: bool
    Whether to sanitize molecules.
  tasks: list, optional (default [])
    Each entry in `tasks` is treated as a property in the SDF file and is
    retrieved with `mol.GetProp(str(task))` where `mol` is the RDKit mol
    loaded from a given SDF entry.
  shard_size: int, optional (default None) 
    The shard size to yield at one time.

  Note
  ----
  This function requires RDKit to be installed.

  Returns
  -------
  dataframes: list
    This function returns a list of pandas dataframes. Each dataframe will
    contain columns `('mol_id', 'smiles', 'mol')`.
  """
  from rdkit import Chem
  df_rows = []
  for input_file in input_files:
    # Tasks are either in .sdf.csv file or in the .sdf file itself
    has_csv = os.path.isfile(input_file + ".csv")
    # Structures are stored in .sdf file
    logger.info("Reading structures from %s." % input_file)
    suppl = Chem.SDMolSupplier(str(input_file), clean_mols, False, False)
    for ind, mol in enumerate(suppl):
      if mol is None:
        continue
      smiles = Chem.MolToSmiles(mol)
      df_row = [ind, smiles, mol]
      if not has_csv:  # Get task targets from .sdf file
        for task in tasks:
          df_row.append(mol.GetProp(str(task)))
      df_rows.append(df_row)
      if shard_size is not None and len(df_rows) == shard_size:
        if has_csv:
          mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol'))
          raw_df = next(load_csv_files([input_file + ".csv"], shard_size=None))
          yield pd.concat([mol_df, raw_df], axis=1, join='inner')
        else:
          mol_df = pd.DataFrame(
              df_rows, columns=('mol_id', 'smiles', 'mol') + tuple(tasks))
          yield mol_df
        # Reset aggregator
        df_rows = []
    # Handle final leftovers for this file
    if len(df_rows) > 0:
      if has_csv:
        mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol'))
        raw_df = next(load_csv_files([input_file + ".csv"], shard_size=None))
        yield pd.concat([mol_df, raw_df], axis=1, join='inner')
      else:
        mol_df = pd.DataFrame(
            df_rows, columns=('mol_id', 'smiles', 'mol') + tuple(tasks))
        yield mol_df
      df_rows = []


def load_csv_files(filenames: List[str],
                   shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
  """Load data as pandas dataframe.

  Parameters
  ----------
  filenames: list[str]
    List of filenames
  shard_size: int, optional (default None) 
    The shard size to yield at one time.

  Returns
  -------
  Iterator which iterates over shards of data.
  """
  # First line of user-specified CSV *must* be header.
  shard_num = 1
  for filename in filenames:
    if shard_size is None:
      yield pd.read_csv(filename)
    else:
      logger.info("About to start loading CSV from %s" % filename)
      for df in pd.read_csv(filename, chunksize=shard_size):
        logger.info(
            "Loading shard %d of size %s." % (shard_num, str(shard_size)))
        df = df.replace(np.nan, str(""), regex=True)
        shard_num += 1
        yield df


def load_json_files(filenames: List[str],
                    shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
  """Load data as pandas dataframe.

  Parameters
  ----------
  filenames : List[str]
    List of json filenames.
  shard_size : int, optional
    Chunksize for reading json files.

  Yields
  ------
  df : pandas.DataFrame
    Shard of dataframe.

  Notes
  -----
  To load shards from a json file into a Pandas dataframe, the file
    must be originally saved with
  ``df.to_json('filename.json', orient='records', lines=True)``

  """

  shard_num = 1
  for filename in filenames:
    if shard_size is None:
      yield pd.read_json(filename, orient='records', lines=True)
    else:
      logger.info("About to start loading json from %s." % filename)
      for df in pd.read_json(
          filename, orient='records', chunksize=shard_size, lines=True):
        logger.info(
            "Loading shard %d of size %s." % (shard_num, str(shard_size)))
        df = df.replace(np.nan, str(""), regex=True)
        shard_num += 1
        yield df


def seq_one_hot_encode(sequences, letters='ATCGN'):
  """One hot encodes list of genomic sequences.

  Sequences encoded have shape (N_sequences, N_letters, sequence_length, 1).
  These sequences will be processed as images with one color channel.

  Parameters
  ----------
  sequences: np.ndarray
    Array of genetic sequences
  letters: str
    String with the set of possible letters in the sequences.

  Raises
  ------
  ValueError:
    If sequences are of different lengths.

  Returns
  -------
  np.ndarray: Shape (N_sequences, N_letters, sequence_length, 1).
  """
  warnings.warn(
      "This Function has been deprecated and now resides in deepchem.utils.genomics_utils ",
      DeprecationWarning)
  return seq_one_hotencode(sequences, letters=letters)


def encode_fasta_sequence(fname):
  """
  Loads fasta file and returns an array of one-hot sequences.

  Parameters
  ----------
  fname: str
    Filename of fasta file.

  Returns
  -------
  np.ndarray: Shape (N_sequences, 5, sequence_length, 1).
  """
  warnings.warn(
      "This Function has been deprecated and now resides in deepchem.utils.genomics_utils",
      DeprecationWarning)

  return encode_sequence(fname)


def encode_bio_sequence(fname, file_type="fasta", letters="ATCGN"):
  """
  Loads a sequence file and returns an array of one-hot sequences.

  Parameters
  ----------
  fname: str
    Filename of fasta file.
  file_type: str
    The type of file encoding to process, e.g. fasta or fastq, this
    is passed to Biopython.SeqIO.parse.
  letters: str
    The set of letters that the sequences consist of, e.g. ATCG.

  Returns
  -------
  np.ndarray: Shape (N_sequences, N_letters, sequence_length, 1).
  """
  warnings.warn(
      "This Function has been deprecated and now resides in deepchem.utils.genomics_utils ",
      DeprecationWarning)
  return encode_sequence(fname, file_type=file_type, letters=letters)


def load_from_disk(filename):
  """Load a dataset from file."""
  name = filename
  if os.path.splitext(name)[1] == ".gz":
    name = os.path.splitext(name)[0]
  extension = os.path.splitext(name)[1]
  if extension == ".pkl":
    return load_pickle_from_disk(filename)
  elif extension == ".joblib":
    return joblib.load(filename)
  elif extension == ".csv":
    # First line of user-specified CSV *must* be header.
    df = pd.read_csv(filename, header=0)
    df = df.replace(np.nan, str(""), regex=True)
    return df
  elif extension == ".npy":
    return np.load(filename, allow_pickle=True)
  else:
    raise ValueError("Unrecognized filetype for %s" % filename)


def load_sharded_csv(filenames):
  """Load a dataset from multiple files. Each file MUST have same column headers"""
  dataframes = []
  for name in filenames:
    placeholder_name = name
    if os.path.splitext(name)[1] == ".gz":
      name = os.path.splitext(name)[0]
    if os.path.splitext(name)[1] == ".csv":
      # First line of user-specified CSV *must* be header.
      df = pd.read_csv(placeholder_name, header=0)
      df = df.replace(np.nan, str(""), regex=True)
      dataframes.append(df)
    else:
      raise ValueError("Unrecognized filetype for %s" % filename)

  # combine dataframes
  combined_df = dataframes[0]
  for i in range(0, len(dataframes) - 1):
    combined_df = combined_df.append(dataframes[i + 1])
  combined_df = combined_df.reset_index(drop=True)
  return combined_df


def load_pickle_from_disk(filename):
  """Load dataset from pickle file."""
  if ".gz" in filename:
    with gzip.open(filename, "rb") as f:
      df = pickle.load(f)
  else:
    with open(filename, "rb") as f:
      df = pickle.load(f)
  return df


def load_dataset_from_disk(save_dir):
  """Loads MoleculeNet train/valid/test/transformers from disk.

  Expects that data was saved using `save_dataset_to_disk` below. Expects the
  following directory structure for `save_dir`:
  
  save_dir/
    |
    ---> train_dir/
    |
    ---> valid_dir/
    |
    ---> test_dir/
    |
    ---> transformers.pkl

  Parameters
  ----------
  save_dir: str

  Returns
  -------
  loaded: bool
    Whether the load succeeded
  all_dataset: (dc.data.Dataset, dc.data.Dataset, dc.data.Dataset)
    The train, valid, test datasets
  transformers: list of dc.trans.Transformer
    The transformers used for this dataset

  See Also
  --------
  save_dataset_to_disk
  """

  train_dir = os.path.join(save_dir, "train_dir")
  valid_dir = os.path.join(save_dir, "valid_dir")
  test_dir = os.path.join(save_dir, "test_dir")
  if not os.path.exists(train_dir) or not os.path.exists(
      valid_dir) or not os.path.exists(test_dir):
    return False, None, list()
  loaded = True
  train = deepchem.data.DiskDataset(train_dir)
  valid = deepchem.data.DiskDataset(valid_dir)
  test = deepchem.data.DiskDataset(test_dir)
  train.memory_cache_size = 40 * (1 << 20)  # 40 MB
  all_dataset = (train, valid, test)
  with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
    transformers = pickle.load(f)
    return loaded, all_dataset, transformers


def save_dataset_to_disk(save_dir, train, valid, test, transformers):
  """Utility used by MoleculeNet to save train/valid/test datasets.

  This utility function saves a train/valid/test split of a dataset along
  with transformers in the same directory. The saved datasets will take the
  following structure:
  
  save_dir/
    |
    ---> train_dir/
    |
    ---> valid_dir/
    |
    ---> test_dir/
    |
    ---> transformers.pkl

  Parameters
  ----------
  save_dir: str
    Filename of directory to save datasets to.
  train: DiskDataset
    Training dataset to save.
  valid: DiskDataset
    Validation dataset to save.
  test: DiskDataset
    Test dataset to save.
  transformers: List
    List of transformers to save to disk.

  See Also
  --------
  load_dataset_from_disk 
  """
  train_dir = os.path.join(save_dir, "train_dir")
  valid_dir = os.path.join(save_dir, "valid_dir")
  test_dir = os.path.join(save_dir, "test_dir")
  train.move(train_dir)
  valid.move(valid_dir)
  test.move(test_dir)
  with open(os.path.join(save_dir, "transformers.pkl"), 'wb') as f:
    pickle.dump(transformers, f)
  return None
