"""
Simple utils to save and load from disk.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

# TODO(rbharath): Use standard joblib once old-data has been regenerated.
import joblib
import scipy
import math
from sklearn.externals import joblib as old_joblib
import gzip
import pickle
import pandas as pd
import numpy as np
import os
import deepchem
from rdkit import Chem
import time


def log(string, verbose=True):
  """Print string if verbose."""
  if verbose:
    print(string)

MATMAGICKEY = "_matmagickey_"

def save_to_disk(dataset, filename, compress=0):
  """Save a dataset to file."""

  joblib.dump(dataset, filename, compress=compress)


def save_sparse_mats(mat_b, filename):
  print("CALLING SAVE SPARSE MATS")
  res = scipy.sparse.vstack(mat_b)
  items_to_save = [res.data, res.indices, res.indptr, res.shape]
  for idx, ii in enumerate(items_to_save):
    np.save(filename+str(idx), ii, allow_pickle=False)

def load_sparse_mats(filename, max_atoms=23):
  print("LOAD SPARSE START")
  files_to_load = [filename+str(idx)+".npy" for idx in range(4)]
  loaded_items = []

  bgn = time.time()

  for fh in files_to_load:
    obj = np.load(fh, allow_pickle=False)
    loaded_items.append(obj)

  print("LOAD OBJ TIME:", time.time()-bgn)
  bgn = time.time()

  # final = scipy.sparse.csr_matrix(
  #   (loaded_items[0],
  #   loaded_items[1],
  #   loaded_items[2]),
  #   shape=loaded_items[3])

  # print("CSR CONSTRUCT TIME:", time.time()-bgn)
  # bgn = time.time()

  # return np.array(
  #   [loaded_items[0],
  #   loaded_items[1],
  #   loaded_items[2],
  #   loaded_items[3]])
  

  return loaded_items
  
  # final = final.A


  # print("CONVERT_TIME:", time.time()-bgn)
  # bgn = time.time()
  # n_mats = math.ceil(final.shape[0]/max_atoms)
  # all_mats = []

  # for m_idx in range(n_mats):
  #   start = m_idx*max_atoms
  #   end = (m_idx+1)*max_atoms
  #   all_mats.append(final[start:end, :])


  # print("LOAD SPARSE END", time.time()-bgn) 
  # return np.array(all_mats)












# def save_to_disk_np(dataset, filename):
#   """Save a dataset to file."""

#   items = []

#   data = []
#   indices = []
#   indptrs = []
#   shapes = []

#   if len(dataset[0]) == 4:
#     for i,j,k,l in dataset:
#       data.append(i)
#       indices.append(j)
#       indptrs.append(k)
#       shapes.append(l)
#     np.savez(open(filename, "wb"),
#       data=data,
#       indices=indices,
#       indptrs=indptrs,
#       shapes=shapes,
#       allow_pickle=False)

#   else:
#     np.save(open(filename, "wb"), dataset, allow_pickle=False)


# def load_from_disk_np(filename):
#   """Save a dataset to file."""
#   start = time.time()
#   # item = np.load(filename, allow_pickle=False)
#   item = np.load(filename)


#   blobs = []

#   if type(item) == np.lib.npyio.NpzFile:
#     # print("NPZ FOUND")
#     for a,b,c,d in zip(item['data'], item['indices'], item['indptrs'], item['shapes']):
#       blobs.append((a,b,c,d))
#     print("NP LOAD TIME:", time.time()-start)
#     return np.array(blobs)

#   else:
#     return item


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


def load_sdf_files(input_files, clean_mols):
  """Load SDF file into dataframe."""
  dataframes = []
  for input_file in input_files:
    # Tasks are stored in .sdf.csv file
    raw_df = next(load_csv_files([input_file + ".csv"], shard_size=None))
    # Structures are stored in .sdf file
    print("Reading structures from %s." % input_file)
    suppl = Chem.SDMolSupplier(str(input_file), clean_mols, False, False)
    df_rows = []
    for ind, mol in enumerate(suppl):
      if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        df_rows.append([ind, smiles, mol])
    mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol'))
    dataframes.append(pd.concat([mol_df, raw_df], axis=1, join='inner'))
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


def load_from_disk(filename):
  """Load a dataset from file."""
  name = filename
  if MATMAGICKEY in name:
    print("LOADING SPARSE MATS")
    return load_sparse_mats(name)

  if os.path.splitext(name)[1] == ".gz":
    name = os.path.splitext(name)[0]
  if os.path.splitext(name)[1] == ".pkl":
    return load_pickle_from_disk(filename)
  elif os.path.splitext(name)[1] == ".joblib":
    try:
      return joblib.load(filename)
    except KeyError:
      # Try older joblib version for legacy files.
      return old_joblib.load(filename)
    except ValueError:
      return old_joblib.load(filename)
  elif os.path.splitext(name)[1] == ".csv":
    # First line of user-specified CSV *must* be header.
    df = pd.read_csv(filename, header=0)
    df = df.replace(np.nan, str(""), regex=True)
    return df
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

  #combine dataframes
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
  train_dir = os.path.join(save_dir, "train_dir")
  valid_dir = os.path.join(save_dir, "valid_dir")
  test_dir = os.path.join(save_dir, "test_dir")
  if os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(
      test_dir):
    loaded = True
    train = deepchem.data.DiskDataset(train_dir)
    valid = deepchem.data.DiskDataset(valid_dir)
    test = deepchem.data.DiskDataset(test_dir)
    all_dataset = (train, valid, test)
    with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
      transformers = pickle.load(f)
  else:
    loaded = False
    all_dataset = None
    transformers = []
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
