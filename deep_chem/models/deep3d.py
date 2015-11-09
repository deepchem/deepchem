"""
Code for training 3D convolutions.
"""
import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from deep_chem.utils.preprocess import split_dataset
from deep_chem.utils.preprocess import tensor_dataset_to_numpy
from deep_chem.utils.evaluate import eval_model
from deep_chem.utils.evaluate import compute_r2_scores

def fit_3D_convolution(per_task_data, task_types, **training_params):
  """
  Perform stochastic gradient descent for a 3D CNN.
  """
  models = {}
  (train, X_train, y_train, _), _ = per_task_data["all"]
  nb_classes = 2
  models["all"] = train_3D_convolution(X_train, y_train, **training_params)
  return models

def train_3D_convolution(X, y, batch_size=50, nb_epoch=1):
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
