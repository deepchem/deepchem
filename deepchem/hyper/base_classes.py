class HyperparamOpt(object):
  """Abstract superclass for hyperparameter search classes.

  This class is an abstract base class for hyperparameter search
  classes in DeepChem. Hyperparameter search is performed on
  `dc.model.Model` classes. Each hyperparameter object accepts a
  `dc.models.Model` class upon construct. When the `hyperparam_search`
  class is invoked, this class is used to construct many different
  concrete models which are trained on the specified training set and
  evaluated on a given validation set.

  Different subclasses of `HyperparamOpt` differ in the choice of
  strategy for searching the hyperparameter evaluation space. This
  class itself is an abstract superclass and should never be directly
  instantiated.
  """

  def __init__(self, model_class):
    """Initialize Hyperparameter Optimizer.

    Note this is an abstract constructor which should only be used by
    subclasses.

    Example
    -------
    This example shows the type of constructor function expected. 

    >>> import sklearn
    >>> import deepchem as dc
    >>> def rf_model_builder(model_params, model_dir):
          sklearn_model = sklearn.ensemble.RandomForestRegressor(**model_params)
          return dc.models.SklearnModel(sklearn_model, model_dir)

    Parameters
    ----------
    model_class: constructor function.
      This parameter must be constructor function which returns an
      object which is an instance of `dc.model.Model`. This function
      must accept two arguments, `model_params` of type `dict` and
      `model_dir`, a string specifying a path to a model directory.
      See the example.
    """
    if self.__class__.__name__ == "HyperparamOpt":
      raise ValueError(
          "HyperparamOpt is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
      )
    self.model_class = model_class
