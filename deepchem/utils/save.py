"""
Simple utils to save and load from disk.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

# TODO(rbharath): Use standard joblib once old-data has been regenerated.
#import joblib
from sklearn.externals import joblib
import gzip
import pickle
import pandas as pd
import numpy as np

def save_to_disk(dataset, filename):
  """Save a dataset to file."""
  joblib.dump(dataset, filename, compress=0)

def load_from_disk(filename):
  """Load a dataset from file."""
  if ".pkl" in filename:
    return load_pickle_from_disk(filename)
  else:
    return joblib.load(filename)

def load_pickle_from_disk(filename):
  """Load dataset from pickle file."""
  if ".gz" in filename:
    with gzip.open(filename, "rb") as f:
      df = pickle.load(f)
  else:
    with open(filename, "rb") as f:
      df = pickle.load(f)
  return df

def load_pandas_from_disk(filename):
  """Load data as pandas dataframe."""
  if ".csv" not in filename:
    return load_from_disk(filename)
  else:
    # First line of user-specified CSV *must* be header.
    df = pd.read_csv(filename, header=0)
    df = df.replace(np.nan, str(""), regex=True)
    return df
