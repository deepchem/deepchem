"""
Fit model. To be incorporated into Model class.
"""

#from deep_chem.models.model import model_builder
import os
import sys
import numpy as np
import deep_chem.models.deep
from deep_chem.utils.dataset import NumpyDataset
from deep_chem.models import Model
#from deep_chem.utils.preprocess import get_metadata_filename
from deep_chem.utils.dataset import load_sharded_dataset
from deep_chem.utils.preprocess import get_task_type

def get_task_names(metadata_df):
  """
  Extract task names from metadata dataframe.
  """
  _, row = metadata_df.iterrows().next()
  return row['task_names']

def fit_model(model_name, model_params, model_dir, data_dir):
  """Builds model from featurized data."""
  task_type = get_task_type(model_name)
  train = NumpyDataset(os.path.join(data_dir, "train"))

  task_types = {task: task_type for task in train.get_task_names()}
  model_params["data_shape"] = train.get_data_shape()

  model = Model.model_builder(model_name, task_types, model_params)
  model.fit(train)
  Model.save_model(model, model_name, model_dir)

  #metadata_filename = get_metadata_filename(data_dir)
  #metadata_df = load_sharded_dataset(metadata_filename)
  #task_names = get_task_names(metadata_df)

  #This simply loads a sample X tensor and finds its shape.
  #sample_X = load_sharded_dataset(metadata_df.iterrows().next()[1]['X'])[0]
  #model_params['data_shape'] = np.shape(sample_X)

  #train_metadata = metadata_df.loc[metadata_df['split'] =="train"]
  #nb_shards = train_metadata.shape[0]
  '''
  MAX_GPU_RAM = float(691007488/50)
  for i, row in train_metadata.iterrows():
    print("Training on shard %d out of %d" % (i+1, nb_shards))
    X = load_sharded_dataset(row['X-transformed'])
    y = load_sharded_dataset(row['y-transformed'])
    w = load_sharded_dataset(row['w'])

    if sys.getsizeof(X) > MAX_GPU_RAM:
      nb_block = float(sys.getsizeof(X))/MAX_GPU_RAM
      nb_sample = np.shape(X)[0]
      interval_points = np.linspace(0,nb_sample,nb_block+1).astype(int)
      for j in range(0,len(interval_points)-1):
        indices = range(interval_points[j],interval_points[j+1])
        X_batch = X[indices,:]
        y_batch = y[indices]
        w_batch = w[indices]
        model.fit_on_batch(X_batch, y_batch, w_batch)
    else:
      model.fit_on_batch(X, y, w)
    '''

