"""
Contains class for gaussian process hyperparameter optimizations.
"""
import logging
import numpy as np
import tempfile
import os
import deepchem
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.utils.evaluate import Evaluator

logger = logging.getLogger(__name__)


def _convert_hyperparam_dict_to_filename(hyper_params):
  """Helper function that converts a dictionary of hyperparameters to a string that can be a filename.

  Parameters
  ----------
  hyper_params: dict
    Maps string of hyperparameter name to int/float/list.

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
    else:
      filename += "_%.2f" % value
  return filename


def compute_parameter_range(params_dict, search_range):
  """Convenience Function to compute parameter search space.

  Parameters
  ----------
  params_dict: dict
    Dictionary mapping strings to Ints/Floats/Lists. For those
    parameters in which int/float is specified, an explicit list of
    parameters is computed with `search_range`.
  search_range: int(float) (default 4)
    For int/float values in `params_dict`, computes optimization range
    on `[initial values / search_range, initial values *
    search_range]`

  Returns
  -------
  param_range: dict 
    Dictionary mapping hyperparameter names to tuples. Each tuple is
    of form `(value_type, value_range)` where `value_type` is a string
    that is either "int" or "cont" and `value_range` is a list of two
    elements of the form `[low, hi]`
  """
  # Range of optimization
  param_range = {}
  for hp, value in params_dict.items():
    if isinstance(value, int):
      value_range = [value // search_range, value * search_range]
      param_range[hp] = ("int", value_range)
      pass
    elif isinstance(value, float):
      value_range = [value / search_range, value * search_range]
      param_range[hp] = ("cont", value_range)
      pass
    return param_range


class GaussianProcessHyperparamOpt(HyperparamOpt):
  """
  Gaussian Process Global Optimization(GPGO)

  This class uses Gaussian Process optimization to select
  hyperparameters. Underneath the hood it uses pyGPGO to optimize
  models. If you don't have pyGPGO installed, you won't be able to use
  this class.

  Note that `params_dict` has a different semantics than for
  `GridHyperparamOpt`. `param_dict[hp]` must be an int/float and is
  used as the center of a search range.

  Example
  -------
  This example shows the type of constructor function expected. 

  >>> import sklearn
  >>> import deepchem as dc
  >>> def rf_model_builder(**model_params):
  ...   rf_params = {k: v for (k, v) in model_params.items() if k != 'model_dir'}
  ...   model_dir = model_params['model_dir']
  ...   sklearn_model = sklearn.ensemble.RandomForestRegressor(**rf_params)
  ...   return dc.models.SklearnModel(sklearn_model, model_dir)
  >>> optimizer = dc.hyper.GaussianProcessHyperparamOpt(rf_model_builder)

  """

  def hyperparam_search(self,
                        params_dict,
                        train_dataset,
                        valid_dataset,
                        transformers,
                        metric,
                        use_max=True,
                        logdir=None,
                        max_iter=20,
                        search_range=4,
                        logfile=None):
    """Perform hyperparameter search using a gaussian process.

    Parameters
    ----------
    params_dict: dict
      Maps hyperparameter names (strings) to possible parameter
      values. The semantics of this list are different than for
      `GridHyperparamOpt`. `params_dict[hp]` must map to an int/float,
      which is used as the center of a search with radius
      `search_range`.
    train_dataset: `dc.data.Dataset`
      dataset used for training
    valid_dataset: `dc.data.Dataset`
      dataset used for validation(optimization on valid scores)
    transformers: list[dc.trans.Transformer]
      transformers for evaluation
    metric: `dc.metrics.Metric`
      metric used for evaluation
    use_max: bool, (default True)
      maximization(True) or minimization(False)
    logdir: str, optional
      The directory in which to store created models. If not set, will
      use a temporary directory.
    max_iter: int, (default 20)
      number of optimization trials
    search_range: int(float) (default 4)
      optimization on [initial values / search_range,
                       initial values * search_range]
      names of parameters that should not be optimized
    logfile: str
      Name of logfile to write results to. If specified, this is must
      be a valid file. If not specified, results of hyperparameter
      search will be written to `logdir/.txt`.


    Returns
    -------
    `(best_model, best_hyperparams, all_scores)` where `best_model` is
    an instance of `dc.model.Models`, `best_hyperparams` is a
    dictionary of parameters, and `all_scores` is a dictionary mapping
    string representations of hyperparameter sets to validation
    scores.
    """
    # Specify logfile
    if logfile:
      log_file = logfile
    elif logdir is not None:
      log_file = os.path.join(logdir, "results.txt")
    else:
      log_file = None

    # setup range
    param_range = compute_parameter_range(params_dict, search_range)
    param_keys = list(param_range.keys())

    # Stores all results
    all_results = {}
    # Stores all model locations
    model_locations = {}

    # Demarcating internal function for readability
    ########################
    def optimizing_function(**placeholders):
      """Private Optimizing function

      Take in hyper parameter values and return valid set performances

      Parameters
      ----------
      placeholders: keyword arguments
        Should be various hyperparameters as specified in `param_keys` above.

      Returns:
      --------
      valid_scores: float
        valid set performances
      """
      hyper_parameters = {}
      for hp in param_keys:
        if param_range[hp][0] == "int":
          # param values are always float in BO, so this line converts float to int
          # see : https://github.com/josejimenezluna/pyGPGO/issues/10
          hyper_parameters[hp] = int(placeholders[hp])
        else:
          hyper_parameters[hp] = float(placeholders[hp])
      logger.info("Running hyperparameter set: %s" % str(hyper_parameters))
      if log_file:
        # Run benchmark
        with open(log_file, 'a') as f:
          # Record hyperparameters
          f.write(str(hyper_parameters))
          f.write('\n')

      hp_str = _convert_hyperparam_dict_to_filename(hyper_parameters)
      if logdir is not None:
        filename = "model%s" % hp_str
        model_dir = os.path.join(logdir, filename)
        logger.info("model_dir is %s" % model_dir)
        try:
          os.makedirs(model_dir)
        except OSError:
          if not os.path.isdir(model_dir):
            logger.info("Error creating model_dir, using tempfile directory")
            model_dir = tempfile.mkdtemp()
      else:
        model_dir = tempfile.mkdtemp()
      # Add it on to the information needed for the constructor
      hyper_parameters["model_dir"] = model_dir
      model = self.model_builder(**hyper_parameters)
      model.fit(train_dataset)
      try:
        model.save()
      # Some models autosave
      except NotImplementedError:
        pass

      evaluator = Evaluator(model, valid_dataset, transformers)
      multitask_scores = evaluator.compute_model_performance([metric])
      score = multitask_scores[metric.name]

      if log_file:
        with open(log_file, 'a') as f:
          # Record performances
          f.write(str(score))
          f.write('\n')
      # Store all results
      all_results[hp_str] = score
      model_locations[hp_str] = model_dir
      # GPGO maximize performance by default, set performance to its negative value for minimization
      if use_max:
        return score
      else:
        return -score

    ########################

    import pyGPGO
    from pyGPGO.covfunc import matern32
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.surrogates.GaussianProcess import GaussianProcess
    from pyGPGO.GPGO import GPGO
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, optimizing_function, param_range)
    logger.info("Max number of iteration: %i" % max_iter)
    gpgo.run(max_iter=max_iter)

    hp_opt, valid_performance_opt = gpgo.getResult()
    hyper_parameters = {}
    for hp in param_keys:
      if param_range[hp][0] == "int":
        hyper_parameters[hp] = int(hp_opt[hp])
      else:
        hyper_parameters[hp] = float(hp_opt[hp])
    hp_str = _convert_hyperparam_dict_to_filename(hyper_parameters)

    # Let's reinitialize the model with the best parameters
    model_dir = model_locations[hp_str]
    hyper_parameters["model_dir"] = model_dir
    best_model = self.model_builder(**hyper_parameters)
    # Some models need to be explicitly reloaded
    try:
      best_model.restore()
    # Some models auto reload
    except NotImplementedError:
      pass

    # Compare best model to default hyperparameters
    if log_file:
      with open(log_file, 'a') as f:
        # Record hyperparameters
        f.write(str(params_dict))
        f.write('\n')

    # Return default hyperparameters
    return best_model, hyper_parameters, all_results
