
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

  def hyperparam_search(self,
                        params_dict,
                        train_dataset,
                        valid_dataset,
                        transformers,
                        metric,
                        use_max=True,
                        logdir=None):
    """Conduct Hyperparameter search.

    This method defines the common API shared by all hyperparameter
    optimization subclasses. Different classes will implement
    different search methods but they must all follow this common API.

    Parameters
    ----------
    params_dict: dict
      Dictionary mapping strings to Ints/Floats/Lists. Note that the
      precise semantics of `params_dict` will change depending on the
      optimizer that you're using.
    train_dataset: `dc.data.Dataset`
      dataset used for training
    valid_dataset: `dc.data.Dataset`
      dataset used for validation(optimization on valid scores)
    output_transformers: list[dc.trans.Transformer]
      Transformers for evaluation. This argument is needed since
      `train_dataset` and `valid_dataset` may have been transformed
      for learning and need the transform to be inverted before
      the metric can be evaluated on a model.
    use_max: bool, optional
      If True, return the model with the highest score. Else return
      model with the minimum score.
    logdir: str, optional
      The directory in which to store created models. If not set, will
      use a temporary directory.

    Returns
    -------
    `(best_model, best_hyperparams, all_scores)` where `best_model` is
    an instance of `dc.model.Models`, `best_hyperparams` is a
    dictionary of parameters, and `all_scores` is a dictionary mapping
    string representations of hyperparameter sets to validation
    scores.
    """
    raise NotImplementedError
