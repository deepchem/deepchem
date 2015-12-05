"""
Utility functions to save keras/sklearn models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gzip
import cPickle as pickle
from keras.models import model_from_json
from sklearn.externals import joblib

def save_model(models, modeltype, filename):
  """Dispatcher function for saving."""
  if modeltype == "sklearn":
    save_sklearn_model(models, filename)
  elif "keras" in modeltype:
    save_keras_model(models, filename)
  else:
    raise ValueError("Unsupported modeltype.")

def save_sharded_dataset(dataset, filename):
  """Save a dataset to file."""
  joblib.dump(dataset, filename, compress=0)

def load_sharded_dataset(filename):
  """Load a dataset from file."""
  dataset = joblib.load(filename)
  return dataset

def load_model(modeltype, filename):
  """Dispatcher function for loading."""
  if modeltype == "sklearn":
    return load_sklearn_model(filename)
  elif "keras" in modeltype:
    return load_keras_model(filename)
  else:
    raise ValueError("Unsupported modeltype.")

def save_sklearn_model(models, filename):
  """Saves sklearn model to disk using joblib."""
  joblib.dump(models, filename)

def load_sklearn_model(filename):
  """Loads sklearn model from file on disk."""
  return joblib.load(filename)

def save_keras_model(models, filename):
  """Saves keras models to disk."""
  filename, _ = os.path.splitext(filename)
  pkl_gz_filename = "%s.%s" % (filename, "pkl.gz")
  with gzip.open(pkl_gz_filename, "wb") as file_obj:
    pickle.dump(models.keys(), file_obj)
  for target in models:
    model = models[target]
    # Note that keras requires the model architecture and weights to be stored
    # separately. A json file is generated that specifies the model architecture.
    # The weights will be stored in an h5 file. The pkl.gz file with store the
    # target name.
    json_filename = "%s-%s.%s" % (filename, target, "json")
    h5_filename = "%s-%s.%s" % (filename, target, "h5")
    # Save architecture
    json_string = model.to_json()
    with open(json_filename, "wb") as file_obj:
      file_obj.write(json_string)
    model.save_weights(h5_filename, overwrite=True)

def load_keras_model(filename):
  """Loads keras model from disk.

  Assumes that filename.json and filename.h5 respectively contain the model
  architecture and weights.
  """
  filename, _ = os.path.splitext(filename)
  pkl_gz_filename = "%s.%s" % (filename, "pkl.gz")
  with gzip.open(pkl_gz_filename) as file_obj:
    targets = pickle.load(file_obj)
  models = {}
  for target in targets:
    json_filename = "%s-%s.%s" % (filename, target, "json")
    h5_filename = "%s-%s.%s" % (filename, target, "h5")

    with open(json_filename) as file_obj:
      model = model_from_json(file_obj.read())
    model.load_weights(h5_filename)
    models[target] = model
  return models
