"""
Utility functions to save models.
"""
from sklearn.externals import joblib

# TODO(rbharath): This implementation only supports saving single models. Make
# some way to save metadata in addition to the actual model file.
def save_model(model, modeltype, filename):
  """Dispatcher function for saving."""
  if modeltype == "sklearn":
    save_sklearn_model(model, filename)
  elif modeltype == "keras":
    save_keras_model(model, filename)
  else:
    raise ValueError("Unsupported modeltype.")

def load_model(modeltype, filename):
  """Dispatcher function for loading."""
  if modeltype == "sklearn":
    return load_sklearn_model(filename)
  elif modeltype == "keras":
    return load_keras_model(filename)
  else:
    raise ValueError("Unsupported modeltype.")

def save_sklearn_model(model, filename):
  """Saves sklearn model to disk using joblib."""
  joblib.dump(model, filename)

def load_sklearn_model(filename):
  """Loads sklearn model from file on disk."""
  return joblib.load(filename)
  
