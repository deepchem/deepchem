"""
Code for training 3D convolutions.
"""
import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from deep_chem.utils.preprocess import train_test_random_split
from deep_chem.utils.load import load_and_transform_dataset
from deep_chem.utils.preprocess import tensor_dataset_to_numpy
from deep_chem.datasets.shapes_3d import load_data
from deep_chem.utils.evaluate import eval_model
from deep_chem.utils.evaluate import compute_r2_scores

# TODO(rbharath): Factor this out into a separate function in utils. Duplicates
# code in deep.py
# TODO(rbharath): paths is to handle sharded input pickle files. Might be
# better to use hdf5 datasets like in MSMBuilder
def process_3D_convolutions(paths, task_transforms, seed=None, splittype="random"):
  """Loads 3D Convolution datasets.

  Parameters
  ----------
  paths: list
    List of paths to convolution datasets.
  """
  dataset = load_and_transform_dataset(paths, task_transforms, datatype="pdbbind")
  # TODO(rbharath): Factor this code splitting out into a util function.
  if splittype == "random":
    train, test = train_test_random_split(dataset, seed=seed)
  elif splittype == "scaffold":
    train, test = train_test_scaffold_split(dataset)
  X_train, y_train, W_train = tensor_dataset_to_numpy(train)
  X_test, y_test, W_test = tensor_dataset_to_numpy(test)
  return (X_train, y_train, W_train, train), (X_test, y_test, W_test, test)

def fit_3D_convolution(paths, task_types, task_transforms, axis_length=32, **training_params):
  """
  Perform stochastic gradient descent for a 3D CNN.
  """
  (X_train, y_train, W_train, train), (X_test, y_test, W_test, test) = process_3D_convolutions(
    paths, task_transforms)

  print "np.shape(X_train): " + str(np.shape(X_train))
  print "np.shape(y_train): " + str(np.shape(y_train))

  nb_classes = 2
  model = train_3D_convolution(X_train, y_train, axis_length, **training_params)
  results = eval_model(test, model, task_types,
      modeltype="keras", mode="tensor")
  local_task_types = task_types.copy()
  r2s = compute_r2_scores(results, local_task_types)
  print "Mean R^2: %f" % np.mean(np.array(r2s.values()))

def train_3D_convolution(X, y, axis_length=32, batch_size=50, nb_epoch=1):
  """
  Fit a keras 3D CNN to datat.

  Parameters
  ----------
  nb_epoch: int
    maximal number of epochs to run the optimizer
  """
  print "Training 3D model"
  print "Original shape of X: " + str(np.shape(X))
  print "Shuffling X dimensions to match convnet"
  # TODO(rbharath): Modify the featurization so that it matches desired shaped. 
  (n_samples, axis_length, _, _, n_channels) = np.shape(X)
  X = np.reshape(X, (n_samples, axis_length, n_channels, axis_length, axis_length))
  print "Final shape of X: " + str(np.shape(X))
  # Number of classes for classification
  nb_classes = 2

  # number of convolutional filters to use at each layer
  nb_filters = [axis_length/2, axis_length, axis_length]

  # level of pooling to perform at each layer (POOL x POOL)
  nb_pool = [2, 2, 2]

  # level of convolution to perform at each layer (CONV x CONV)
  nb_conv = [7, 5, 3]

  model = Sequential()
  model.add(Convolution3D(nb_filter=nb_filters[0], stack_size=n_channels,
     nb_row=nb_conv[0], nb_col=nb_conv[0], nb_depth=nb_conv[0],
     border_mode='valid'))
  model.add(Activation('relu'))
  model.add(MaxPooling3D(poolsize=(nb_pool[0], nb_pool[0], nb_pool[0])))
  model.add(Convolution3D(nb_filter=nb_filters[1], stack_size=nb_filters[0],
     nb_row=nb_conv[1], nb_col=nb_conv[1], nb_depth=nb_conv[1],
     border_mode='valid'))
  model.add(Activation('relu'))
  model.add(MaxPooling3D(poolsize=(nb_pool[1], nb_pool[1], nb_pool[1])))
  model.add(Convolution3D(nb_filter=nb_filters[2], stack_size=nb_filters[1],
     nb_row=nb_conv[2], nb_col=nb_conv[2], nb_depth=nb_conv[2],
     border_mode='valid'))
  model.add(Activation('relu'))
  model.add(MaxPooling3D(poolsize=(nb_pool[2], nb_pool[2], nb_pool[2])))
  model.add(Flatten())
  # TODO(rbharath): If we change away from axis-size 32, this code will break.
  # Eventually figure out a more general rule that works for all axis sizes.
  model.add(Dense(32, 32/2, init='normal'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # TODO(rbharath): Generalize this to support classification as well as regression.
  model.add(Dense(32/2, 1, init='normal'))

  sgd = RMSprop(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  print "About to compile model"
  model.compile(loss='mean_squared_error', optimizer=sgd)
  print "About to fit data to model."
  model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch)
  return model
