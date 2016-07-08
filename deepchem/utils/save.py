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
import os

def log(string, verbosity=None, level="low"):
  """Print string if verbose."""
  assert level in ["low", "high"]
  if verbosity is not None:
    if verbosity == "high" or level == verbosity:
      print(string)

def save_to_disk(dataset, filename, compress=3):
  """Save a dataset to file."""
  joblib.dump(dataset, filename, compress=compress)

def load_from_disk(filename):
  """Load a dataset from file."""
  name = filename
  if os.path.splitext(name)[1] == ".gz":
    name = os.path.splitext(name)[0]
  if os.path.splitext(name)[1] == ".pkl":
    return load_pickle_from_disk(filename)
  elif os.path.splitext(name)[1] == ".joblib":
    return joblib.load(filename)
  elif os.path.splitext(name)[1] == ".csv":
    # First line of user-specified CSV *must* be header.
    df = pd.read_csv(filename, header=0)
    df = df.replace(np.nan, str(""), regex=True)
    return df
  else:
    raise ValueError("Unrecognized filetype for %s" % filename)

# Only handles *.csv.gz files
def load_twofiles_from_disk(filename1, filename2):
  """Load a dataset from file."""
  name1 = filename1
  name2 = filename2
  filenameList = []
  filenameList.append(name1)
  filenameList.append(name2)
  dataframeList = []
  for name in filenameList:
    placeholderName = name
    if os.path.splitext(name)[1] == ".gz":
      #pandas read_csv() method handles gzipped csv files 
      name = os.path.splitext(name)[0]
    if os.path.splitext(name)[1] == ".csv":
      # First line of user-specified CSV *must* be header.
      df = pd.read_csv(placeholderName, header=0)
      df = df.replace(np.nan, str(""), regex=True)
      dataframeList.append(df)
    else:
      raise ValueError("Unrecognized filetype for %s" % filename)
  combined_df = dataframeList[0].append(dataframeList[1])
  return combined_df
 
def load_multfiles_from_disk(filenameList):
  """Load a dataset from multiple files. Each file MUST have same column headers"""
  dataframeList = []
  for name in filenameList:
    placeholderName = name
    if os.path.splitext(name)[1] == ".gz":
      name = os.path.splitext(name)[0]
    if os.path.splitext(name)[1] == ".csv":
      # First line of user-specified CSV *must* be header.
      df = pd.read_csv(placeholderName, header=0)
      df = df.replace(np.nan, str(""), regex=True)
      dataframeList.append(df)
    else:
      raise ValueError("Unrecognized filetype for %s" % filename)
  
  #combine dataframes
  combined_df = dataframeList[0]
  for i in range(0, len(dataframeList) - 1):
    combined_df = combined_df.append(dataframeList[i+1])
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

