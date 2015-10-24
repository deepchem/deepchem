"""
Code for training 3D convolutions.
"""
from deep_chem.datasets.shapes_3d import load_data
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.utils import np_utils
import numpy as np

# TODO(rbharath): Factor this out into a separate function in utils. Duplicates
# code in deep.py
def process_3D_convolutions(paths, task_transforms, splittype="random"):
  """Loads 3D Convolution datasets.

  Parameters
  ----------
  paths: list
    List of paths to convolution datasets.
  """
  dataset = load_and_transform_dataset(paths, task_transforms)
  # TODO(rbharath): Factor this code splitting out into a util function.
  if splittype == "random":
    train, test = train_test_random_split(dataset, seed=seed)
  elif splittype == "scaffold":
    train, test = train_test_scaffold_split(dataset)
  X_train, y_train, W_train = dataset_to_numpy(train)
  X_test, y_test, W_test = dataset_to_numpy(test)
  return (X_train, y_train, W_train), (X_test, y_test, W_test)

def fit_3D_convolution(axis_length=32, **training_params):
  """
  Perform stochastic gradient descent for a 3D CNN.
  """
  nb_classes = 2
  (X_train, y_train), (X_test, y_test) = load_data(axis_length=axis_length)
  y_train = np_utils.to_categorical(y_train, nb_classes)
  y_test = np_utils.to_categorical(y_test, nb_classes)
  print "np.shape(X_train): " + str(np.shape(X_train))
  print "np.shape(y_train): " + str(np.shape(y_train))
  train_3D_convolution(X_train, y_train, axis_length, **training_params)

def train_3D_convolution(X, y, axis_length=32, batch_size=50, nb_epoch=1):
  """
  Fit a keras 3D CNN to datat.

  Parameters
  ----------
  nb_epoch: int
    maximal number of epochs to run the optimizer
  """
  print "train_3D_convolution"
  print "axis_length: " + str(axis_length)
  # Number of classes for classification
  nb_classes = 2

  # number of convolutional filters to use at each layer
  nb_filters = [axis_length/2, axis_length, axis_length]
  print "nb_filters: " + str(nb_filters)

  # level of pooling to perform at each layer (POOL x POOL)
  nb_pool = [2, 2, 2]

  # level of convolution to perform at each layer (CONV x CONV)
  nb_conv = [7, 5, 3]

  model = Sequential()
  model.add(Convolution3D(nb_filter=nb_filters[0], stack_size=1,
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
  model.add(Dense(320, 32/2, init='normal'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(32/2, nb_classes, init='normal'))
  model.add(Activation('softmax'))

  sgd = RMSprop(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch)

  return model
