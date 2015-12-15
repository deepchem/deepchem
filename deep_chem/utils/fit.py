from deep_chem.models.model import model_builder
from deep_chem.utils.preprocess import get_metadata_filename
from deep_chem.utils.save import load_sharded_dataset
import pandas as pd
from deep_chem.preprocess import get_task_type

def get_task_names(metadata_df):
  row = metadata_df.iterrows().next()
  return(row['task_names'])

def fit_model(model_type, model_params, saved_out, data_dir):
  """Builds model from featurized data."""
  task_type = get_task_type(model_type)
  model = model_builder(model_type, task_types, model_params)
  metadata_filename = get_metadata_filename(data_dir)
  metadata_df = load_sharded_dataset(metadata_filename)
  task_names = get_task_names(metadata_df)
  task_types = {task: task_type for task in task_names} 

  for row in metadata_df.iterrows():
    if row['split'] != "train":
      continue 

    X = load_sharded_dataset(row['X'])
    y = load_sharded_dataset(row['y'])
    w = load_sharded_dataset(row['w'])

    model.train_on_batch(X, y, w)

  save_model(models, modeltype, saved_out)
