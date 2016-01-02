"""
Code for processing the Google vs-datasets using keras.
"""
import numpy as np
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from deep_chem.models import Model

class MultiTaskDNN(Model):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, task_types, model_params, initialize_raw_model=True):
    super(MultiTaskDNN, self).__init__(task_types, model_params,
                                       initialize_raw_model)
    if initialize_raw_model:
      sorted_tasks = sorted(task_types.keys())
      (n_inputs,) = model_params["data_shape"]
      model = Graph()
      model.add_input(name="input", input_shape=(n_inputs,))
      model.add_node(
          Dense(model_params["nb_hidden"], init='uniform',
                activation=model_params["activation"]),
          name="dense", input="input")
      model.add_node(Dropout(model_params["dropout"]), name="dropout",
                             input="dense")
      top_layer = "dropout"
      for ind, task in enumerate(sorted_tasks):
        task_type = task_types[task]
        if task_type == "classification":
          model.add_node(
              Dense(2, init='uniform', activation="softmax"),
              name="dense_head%d" % ind, input=top_layer)
        elif task_type == "regression":
          model.add_node(
              Dense(1, init='uniform'),
              name="dense_head%d" % ind, input=top_layer)
        model.add_output(name="task%d" % ind, input="dense_head%d" % ind)

      loss_dict = {}
      for ind, task in enumerate(sorted_tasks):
        task_type, taskname = task_types[task], "task%d" % ind
        if task_type == "classification":
          loss_dict[taskname] = "binary_crossentropy"
        elif task_type == "regression":
          loss_dict[taskname] = "mean_squared_error"
      sgd = SGD(lr=model_params["learning_rate"],
                decay=model_params["decay"],
                momentum=model_params["momentum"],
                nesterov=model_params["nesterov"])
      model.compile(optimizer=sgd, loss=loss_dict)
      self.raw_model = model

  def get_data_dict(self, X, y=None):
    data = {}
    data["input"] = X
    for ind, task in enumerate(sorted(self.task_types.keys())):
      task_type, taskname = task_types[task], "task%d" % ind
      if y is not None:
        if task_type == "classification":
          data[taskname] = to_one_hot(y[:, ind])
        elif task_type == "regression":
          data[taskname] = y[:, ind]
    return data

  def get_sample_weight(self, w):
    """Get dictionaries needed to fit models"""
    sample_weight = {}
    for ind, task in enumerate(sorted(self.task_types.keys())):
      sample_weight["task%d" % ind] = w[:, ind]
    return sample_weight

  def fit_on_batch(self, X, y, w):
    """
    Updates existing model with new information.
    """
    eps = .001
    # Add eps weight to avoid minibatches with zero weight (causes theano to crash).
    W = W + eps * np.ones(np.shape(W))
    data = self.get_data_dict(X, y)
    sample_weight = self.get_sample_weight(w)
    loss = self.raw_model.train_on_batch(data, sample_weight=sample_weight)

  def predict_on_batch(self, X):
    """
    Makes predictions on given batch of new data.
    """
    data = self.get_data_dict(X)
    y_pred = self.raw_model.predict_on_batch(data)
    y_pred = np.squeeze(y_pred)
    return y_pred

Model.register_model_type("multitask_deep_regressor", MultiTaskDNN)
Model.register_model_type("multitask_deep_classifier", MultiTaskDNN)

class SingleTaskDNN(MultiTaskDNN):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, task_types, model_params, initialize_raw_model=True):
    super(SingleTaskDNN, self).__init__(task_types, model_params,
                                       initialize_raw_model)

Model.register_model_type("singletask_deep_regressor", SingleTaskDNN)
Model.register_model_type("singletask_deep_classifier", SingleTaskDNN)

def to_one_hot(y):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape [n_samples, 2] (assuming binary labels).

  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, 2))
  for index, val in enumerate(y):
    if val == 0:
      y_hot[index] = np.array([1, 0])
    elif val == 1:
      y_hot[index] = np.array([0, 1])
  return y_hot

def fit_multitask_mlp(train_data, task_types, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras multitask MLP.
  Returns AUCs, R^2 scores, and RMS values.

  Parameters
  ----------
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  models = {}
  # Follows convention from process_datasets that the data for multitask models
  # is grouped under key "all"
  X_train = train_data["features"]
  (y_train, W_train) = train_data["all"]
  models["all"] = train_multitask_model(X_train, y_train, W_train, task_types,
                                        **training_params)
  return models

def fit_singletask_mlp(train_data, task_types, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras MLP.

  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  output_transforms: dict
    dict mapping task names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  models = {}
  train_ids = train_data["mol_ids"]
  X_train = train_data["features"]
  sorted_tasks = train_data["sorted_tasks"]
  for index, task in enumerate(sorted_tasks):
    print "Training model %d" % index
    print "Target %s" % task
    (y_train, W_train) = train_data[task]
    flat_W_train = W_train.ravel()
    task_X_train = X_train[flat_W_train.nonzero()]
    task_y_train = y_train[flat_W_train.nonzero()]
    print "%d compounds in Train" % len(train_ids)
    models[task] = train_multitask_model(task_X_train, task_y_train, W_train,
                                         {task: task_types[task]},
                                         **training_params)
  return models

def train_multitask_model(X, y, W, task_types, learning_rate=0.01,
                          decay=1e-6, momentum=0.9, nesterov=True, activation="relu",
                          dropout=0.5, nb_epoch=20, batch_size=50, nb_hidden=500,
                          validation_split=0.1):
  """
  Perform stochastic gradient descent optimization for a keras multitask MLP.
  Returns a trained model.

  Parameters
  ----------
  X: np.ndarray
    Feature matrix
  y: np.ndarray
    Label matrix
  W: np.ndarray
    Weight matrix
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  learning_rate: float
    Learning rate used.
  decay: float
    Learning rate decay.
  momentum: float
    Momentum used in SGD.
  nesterov: bool
    Use Nesterov acceleration
  nb_epoch: int
    maximal number of epochs to run the optimizer
  """
  print "Done compiling. About to fit model!"
  print "validation_split: " + str(validation_split)
  model.fit(data_dict, nb_epoch=nb_epoch, batch_size=batch_size,
            validation_split=validation_split, sample_weight=sample_weights)
  return model
