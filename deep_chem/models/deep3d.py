"""
Code for training 3D convolutions.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from deep_chem.models import Model

class DockingDNN(Model):
  """
  Wrapper class for fitting 3D convolutional networks for deep docking.
  """
  def __init__(self, task_types, model_params, initialize_raw_model=True):
    if initialize_raw_model:
      (axis_length, _, _, n_channels) = model_params["data_shape"]

      learning_rate = model_params["learning_rate"]
      loss_function = model_params["loss_function"]

         # number of convolutional filters to use at each layer
      nb_filters = [axis_length/2, axis_length, axis_length]

      # level of pooling to perform at each layer (POOL x POOL)
      nb_pool = [2, 2, 2]

      # level of convolution to perform at each layer (CONV x CONV)
      nb_conv = [7, 5, 3]
      model = Sequential()
      model.add(Convolution3D(nb_filter=nb_filters[0], stack_size=n_channels,
                              nb_row=nb_conv[0], nb_col=nb_conv[0],
                              nb_depth=nb_conv[0], border_mode='valid'))
      model.add(Activation('relu'))
      model.add(MaxPooling3D(poolsize=(nb_pool[0], nb_pool[0], nb_pool[0])))
      model.add(Convolution3D(nb_filter=nb_filters[1], stack_size=nb_filters[0],
                              nb_row=nb_conv[1], nb_col=nb_conv[1], nb_depth=nb_conv[1],
                              border_mode='valid'))
      model.add(Activation('relu'))
      model.add(MaxPooling3D(poolsize=(nb_pool[1], nb_pool[1], nb_pool[1])))
      model.add(Convolution3D(nb_filter=nb_filters[2], stack_size=nb_filters[1],
                              nb_row=nb_conv[2], nb_col=nb_conv[2],
                              nb_depth=nb_conv[2], border_mode='valid'))
      model.add(Activation('relu'))
      model.add(MaxPooling3D(poolsize=(nb_pool[2], nb_pool[2], nb_pool[2])))
      model.add(Flatten())
      # TODO(rbharath): If we change away from axis-size 32, this code will break.
      # Eventually figure out a more general rule that works for all axis sizes.
      model.add(Dense(32/2, init='normal'))
      model.add(Activation('relu'))
      model.add(Dropout(0.5))
      # TODO(rbharath): Generalize this to support classification as well as regression.
      model.add(Dense(1, init='normal'))

      sgd = RMSprop(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
      print("About to compile model")
      model.compile(loss=loss_function, optimizer=sgd)
      self.model = model
    super(DockingDNN, self).__init__(task_types, model_params, initialize_raw_model)

  def fit_on_batch(self, X, y, w):
    print("Training 3D model")
    print("Original shape of X: " + str(np.shape(X)))
    print("Shuffling X dimensions to match convnet")
    # TODO(rbharath): Modify the featurization so that it matches desired shaped.
    (n_samples, axis_length, _, _, n_channels) = np.shape(X)
    X = np.reshape(X, (n_samples, axis_length, n_channels, axis_length, axis_length))
    print("Final shape of X: " + str(np.shape(X)))

    print("About to fit data to model.")
    batch_size = self.model_params["batch_size"]
    nb_epoch = self.model_params["nb_epoch"]
    y = y.itervalues().next()
    self.raw_model.train_on_batch(X, y, batch_size=batch_size, nb_epoch=nb_epoch)
    print("Finished training on batch.")

  def predict_on_batch(self, X):
    if len(np.shape(X)) != 5:
      raise ValueError(
          "Tensorial datatype must be of shape (n_samples, N, N, N, n_channels).")
    (n_samples, axis_length, _, _, n_channels) = np.shape(X)
    X = np.reshape(X, (n_samples, axis_length, n_channels, axis_length, axis_length))
    y_pred = self.raw_model.predict_on_batch(X)
    y_pred = np.squeeze(y_pred)
    return(y_pred)
