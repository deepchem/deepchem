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
from deepchem.utils.genomics import encode_bio_sequence as encode_sequence, encode_fasta_sequence as fasta_sequence, seq_one_hot_encode as seq_one_hotencode


def log(string, verbose=True):
  """Print string if verbose."""
  if verbose:
    print(string)


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


def load_data(input_files, shard_size=None, verbose=True):
  """Loads data from disk.

  For CSV files, supports sharded loading for large files.
  """
  if not len(input_files):
    return
  input_type = get_input_type(input_files[0])
  if input_type == "sdf":
    if shard_size is not None:
      log("Ignoring shard_size for sdf input.", verbose)
    for value in load_sdf_files(input_files):
      yield value
  elif input_type == "csv":
    for value in load_csv_files(input_files, shard_size, verbose=verbose):
      yield value
  elif input_type == "pandas-pickle":
    for input_file in input_files:
      yield load_pickle_from_disk(input_file)


def load_sdf_files(input_files, clean_mols, tasks=[]):
  """Load SDF file into dataframe."""
  from rdkit import Chem
  dataframes = []
  for input_file in input_files:
    # Tasks are either in .sdf.csv file or in the .sdf file itself
    has_csv = os.path.isfile(input_file + ".csv")
    # Structures are stored in .sdf file
    print("Reading structures from %s." % input_file)
    suppl = Chem.SDMolSupplier(str(input_file), clean_mols, False, False)
    df_rows = []
    for ind, mol in enumerate(suppl):
      if mol is None:
        continue
      smiles = Chem.MolToSmiles(mol)
      df_row = [ind, smiles, mol]
      if not has_csv:  # Get task targets from .sdf file
        for task in tasks:
          df_row.append(mol.GetProp(str(task)))
      df_rows.append(df_row)
    if has_csv:
      mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol'))
      raw_df = next(load_csv_files([input_file + ".csv"], shard_size=None))
      dataframes.append(pd.concat([mol_df, raw_df], axis=1, join='inner'))
    else:
      mol_df = pd.DataFrame(
          df_rows, columns=('mol_id', 'smiles', 'mol') + tuple(tasks))
      dataframes.append(mol_df)
  return dataframes


def load_csv_files(filenames, shard_size=None, verbose=True):
  """Load data as pandas dataframe."""
  # First line of user-specified CSV *must* be header.
  shard_num = 1
  for filename in filenames:
    if shard_size is None:
      yield pd.read_csv(filename)
    else:
      log("About to start loading CSV from %s" % filename, verbose)
      for df in pd.read_csv(filename, chunksize=shard_size):
        log("Loading shard %d of size %s." % (shard_num, str(shard_size)),
            verbose)
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
      "This Function has been deprecated and now resides in deepchem.utils.genomics ",
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
      "This Function has been deprecated and now resides in deepchem.utils.genomics",
      DeprecationWarning)

  return fasta_sequence(fname)


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
      "This Function has been deprecated and now resides in deepchem.utils.genomics ",
      DeprecationWarning)
  return encode_sequence(fname, file_type=file_type, letters=letters)


def save_metadata(tasks, metadata_df, data_dir):
  """
  Saves the metadata for a DiskDataset
  Parameters
  ----------
  tasks: list of str
    Tasks of DiskDataset
  metadata_df: pd.DataFrame
  data_dir: str
    Directory to store metadata
  Returns
  -------
  """
  if isinstance(tasks, np.ndarray):
    tasks = tasks.tolist()
  metadata_filename = os.path.join(data_dir, "metadata.csv.gzip")
  tasks_filename = os.path.join(data_dir, "tasks.json")
  with open(tasks_filename, 'w') as fout:
    json.dump(tasks, fout)
  metadata_df.to_csv(metadata_filename, index=False, compression='gzip')


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
  """
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
  train_dir = os.path.join(save_dir, "train_dir")
  valid_dir = os.path.join(save_dir, "valid_dir")
  test_dir = os.path.join(save_dir, "test_dir")
  train.move(train_dir)
  valid.move(valid_dir)
  test.move(test_dir)
  with open(os.path.join(save_dir, "transformers.pkl"), 'wb') as f:
    pickle.dump(transformers, f)
  return None
