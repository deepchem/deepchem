class Model:
  def __init__(self, task_types, training_params):
    self.task_types = task_types
    self.training_params = training_params

  def fit_on_batch(self, X, y, w):
    raise NotImplementedError(
        "Each model is responsible for its own fit_on_batch method.")

  def predict_on_batch(self, X):
    raise NotImplementedError(
        "Each model is responsible for its own predict_on_batch method.")    