import logging

logger = logging.getLogger(__name__)


def _convert_hyperparam_dict_to_filename(hyper_params):
  """Helper function that converts a dictionary of hyperparameters to a string that can be a filename.

  Parameters
  ----------
  hyper_params: dict
    Maps string of hyperparameter name to int/float.

  Returns
  -------
  filename: str
    A filename of form "_key1_value1_value2_..._key2..."
  """
  filename = ""
  keys = sorted(hyper_params.keys())
  for key in keys:
    filename += "_%s" % str(key)
    value = hyper_params[key]
    if isinstance(value, int):
      filename += "_%s" % str(value)
    elif isinstance(value, float):
      filename += "_%.2f" % value
    else:
      filename += "%s" % str(value)
  return filename


class HyperparamOpt(object):
  """Abstract superclass for hyperparameter search classes.

  This class is an abstract base class for hyperparameter search
  classes in DeepChem. Hyperparameter search is performed on
  `dc.models.Model` classes. Each hyperparameter object accepts a
  `dc.models.Model` class upon construct. When the `hyperparam_search`
  class is invoked, this class is used to construct many different
  concrete models which are trained on the specified training set and
  evaluated on a given validation set.

  Different subclasses of `HyperparamOpt` differ in the choice of
  strategy for searching the hyperparameter evaluation space. This
  class itself is an abstract superclass and should never be directly
  instantiated.
  """

  def __init__(self, model_builder):
    """Initialize Hyperparameter Optimizer.

    Note this is an abstract constructor which should only be used by
    subclasses.

    Parameters
    ----------
    model_builder: constructor function.
      This parameter must be constructor function which returns an
      object which is an instance of `dc.models.Model`. This function
      must accept two arguments, `model_params` of type `dict` and
      `model_dir`, a string specifying a path to a model directory.
      See the example.
    """
    if self.__class__.__name__ == "HyperparamOpt":
      raise ValueError(
          "HyperparamOpt is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
      )
    self.model_builder = model_builder

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
      Dictionary mapping strings to values. Note that the
      precise semantics of `params_dict` will change depending on the
      optimizer that you're using. Depending on the type of
      hyperparameter optimization, these values can be
      ints/floats/strings/lists/etc. Read the documentation for the
      concrete hyperparameter optimization subclass you're using to
      learn more about what's expected.
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
    an instance of `dc.models.Models`, `best_hyperparams` is a
    dictionary of parameters, and `all_scores` is a dictionary mapping
    string representations of hyperparameter sets to validation
    scores.
    """
    raise NotImplementedError
