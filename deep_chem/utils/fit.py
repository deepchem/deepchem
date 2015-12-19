"""
Fit model. To be incorporated into Model class.
"""

from deep_chem.models.model import model_builder
from deep_chem.utils.preprocess import get_metadata_filename
from deep_chem.utils.save import load_sharded_dataset
from deep_chem.utils.save import save_model
from deep_chem.utils.preprocess import get_task_type
import numpy as np

def get_task_names(metadata_df):
  """
  Extract task names from metadata dataframe.
  """
  _, row = metadata_df.iterrows().next()
  return row['task_names']

def fit_model(model_name, model_params, model_dir, data_dir):
  """Builds model from featurized data."""
  task_type = get_task_type(model_name)
  metadata_filename = get_metadata_filename(data_dir)
  metadata_df = load_sharded_dataset(metadata_filename)
  task_names = get_task_names(metadata_df)
  task_types = {task: task_type for task in task_names}

  #This simply loads a sample X tensor and finds its shape.
  sample_X = load_sharded_dataset(metadata_df.iterrows().next()[1]['X'])[0]
  model_params['data_shape'] = np.shape(sample_X)

  print("model_params")
  print(model_params)

  model = model_builder(model_name, task_types, model_params)

  print("model")
  print(model)

  train_metadata = metadata_df.loc[metadata_df['split'] =="train"]
  nb_batch = train_metadata.shape[0]

  for i, row in train_metadata.iterrows():
    print("Training on batch %d out of %d" % (i+1, nb_batch))

    X = load_sharded_dataset(row['X'])
    y = load_sharded_dataset(row['y'])
    w = load_sharded_dataset(row['w'])

    model.fit_on_batch(X, y, w)

  save_model(model, model_name, model_dir)
