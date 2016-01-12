"""
Simple utils to save and load from disk.
"""
import joblib

def save_to_disk(dataset, filename):
  """Save a dataset to file."""
  joblib.dump(dataset, filename, compress=0)

def load_from_disk(filename):
  """Load a dataset from file."""
  dataset = joblib.load(filename)
  return dataset
