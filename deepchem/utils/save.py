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
import cPickle as pickle

def save_to_disk(dataset, filename):
  """Save a dataset to file."""
  joblib.dump(dataset, filename, compress=0)

def load_from_disk(filename):
  """Load a dataset from file."""
  dataset = joblib.load(filename)
  return dataset

def load_pickle_from_disk(filename):
  """Load dataset from pickle file."""
  if ".gz" in filename:
    with gzip.open(filename, "rb") as f:
      df = pickle.load(f)
  else:
    with open(filename, "rb") as f:
      df = pickle.load(f)
  return df
