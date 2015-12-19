"""
Utility functions to save keras/sklearn models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from keras.models import model_from_json
from sklearn.externals import joblib
from deep_chem.models.model import model_builder

def get_model_type(model_name):
  """Associate each model with a model_type (used for saving/loading)."""
  if model_name in ["singletask_deep_classifier", "multitask_deep_classifier",
                    "singletask_deep_regressor", "multitask_deep_regressor"]:
    model_type = "keras-graph"
  elif model_name in ["convolutional_3D_regressor"]:
    model_type = "keras-sequential"
  elif model_name == "neural_fingerprint":
    model_type = "autograd"
  else:
    model_type = "sklearn"
  return model_type

def get_parameter_filename(model_dir):
  """
  Given model directory, obtain filename for stored parameters.
  """
  filename = os.path.join(model_dir, "model_params.joblib")
  return filename

def get_model_filename(model_dir):
  """
  Given model directory, obtain filename for the model itself.
  """
  filename = os.path.join(model_dir, "model_params.joblib")
  return filename

def save_model(model, model_name, model_dir):
  """Dispatcher function for saving."""
  model_type = get_model_type(model_name)
  params = {"model_params" : model.model_params,
            "task_types" : model.task_types}
  save_sharded_dataset(params, get_parameter_filename(model_dir))

  raw_model = model.get_raw_model()
  if model_type == "sklearn":
    save_sklearn_model(raw_model, get_model_filename(model_dir))
  elif "keras" in model_type:
    save_keras_model(raw_model, get_model_filename(model_dir))
  else:
    raise ValueError("Unsupported model_type.")

def save_sharded_dataset(dataset, filename):
  """Save a dataset to file."""
  joblib.dump(dataset, filename, compress=0)

def load_sharded_dataset(filename):
  """Load a dataset from file."""
  dataset = joblib.load(filename)
  return dataset

def load_model(model_name, model_dir):
  """Dispatcher function for loading."""
  model_type = get_model_type(model_name)
  params = load_sharded_dataset(get_parameter_filename(model_dir))
  model = model_builder(model_name, params["task_types"],
                        params["model_params"], initialize_raw_model=False)
  if model_type == "sklearn":
    raw_model = load_sklearn_model(get_model_filename(model_dir))
  elif "keras" in model_type:
    raw_model = load_keras_model(get_model_filename(model_dir))
  else:
    raise ValueError("Unsupported model_type.")
  model.set_raw_model(raw_model)
  return model

def save_sklearn_model(model, filename):
  """Saves sklearn model to disk using joblib."""
  joblib.dump(model, filename)

def load_sklearn_model(filename):
  """Loads sklearn model from file on disk."""
  return joblib.load(filename)

def save_keras_model(model, filename):
  """Saves keras models to disk."""
  filename, _ = os.path.splitext(filename)

  # Note that keras requires the model architecture and weights to be stored
  # separately. A json file is generated that specifies the model architecture.
  # The weights will be stored in an h5 file. The pkl.gz file with store the
  # target name.
  json_filename = "%s.%s" % (filename, "json")
  h5_filename = "%s.%s" % (filename, "h5")
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

  json_filename = "%s.%s" % (filename, "json")
  h5_filename = "%s.%s" % (filename, "h5")

  with open(json_filename) as file_obj:
    model = model_from_json(file_obj.read())
  model.load_weights(h5_filename)
  return model
