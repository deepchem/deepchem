"""
Code for training 3D convolutions.
"""

def fit_3D_convolution():
  """
  Perform stochastic gradient descent for a 3D CNN.
  """
  pass

def train_3D_convolution(X, y):
  """
  Fit a keras 3D CNN to datat.
  """
  # number of convolutional filters to use at each layer
  nb_filters = [16, 32, 32]

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
  model.add(Dense(32, 16, init='normal'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(16, nb_classes, init='normal'))
  model.add(Activation('softmax'))

  sgd = RMSprop(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch)

  return model
