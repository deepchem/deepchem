"""
Contains class for random hyperparameter optimizations.
"""
import numpy as np
import os
import tempfile
import collections
import logging
import itertools
from typing import Dict, List, Optional, Tuple, Any, Callable

from deepchem.data import Dataset
from deepchem.trans import Transformer
from deepchem.models import Model
from deepchem.metrics import Metric
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename

logger = logging.getLogger(__name__)


class RandomHyperparamOpt(HyperparamOpt):
  """
  Provides simple random hyperparameter search capabilities.

  This class performs a random hyperparameter search over the specified
  hyperparameter space. This implementation is simple and samples
  hyperparameters from the hyperparameter space passed. However it does not
  use parallelization to speed up the search.

  Examples
  --------
  This example shows the type of constructor function expects.

  >>> import sklearn
  >>> import deepchem as dc
  >>> optimizer = dc.hyper.RandomHyperparamOpt(lambda **p: dc.models.GraphConvModel(**p), max_iter=5)

  Here's a more sophisticated example that shows how to optimize only
  some parameters of a model. In this case, we have some parameters we
  want to optimize, and others which we don't. To handle this type of
  search, we create a `model_builder` which hard codes some arguments.

  >>> import deepchem as dc
  >>> import numpy as np
  >>> from sklearn.linear_model import LogisticRegression as LR
  >>> # generating data
  >>> X = np.arange(1, 11, 1).reshape(-1, 1)
  >>> y = np.hstack((np.zeros(5), np.ones(5)))
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>> # splitting dataset into train and test
  >>> splitter = dc.splits.RandomSplitter()
  >>> train_dataset, test_dataset = splitter.train_test_split(dataset)
  >>> # metric to evaluate result of a set of parameters
  >>> metric = dc.metrics.Metric(dc.metrics.accuracy_score)
  >>> # defining `model_builder`
  >>> def model_builder(**model_params):
  ...   penalty = model_params['penalty']
  ...   solver = model_params['solver']
  ...   lr = LR(penalty=penalty, solver=solver, max_iter=100)
  ...   return dc.models.SklearnModel(lr)
  >>> # the parameters which are to be optimized
  >>> params = {
  ...   'penalty': ['l1', 'l2'],
  ...   'solver': ['liblinear', 'saga']
  ...   }
  >>> # Creating optimizer and searching over hyperparameters
  >>> optimizer = dc.hyper.RandomHyperparamOpt(model_builder, max_iter=100)
  >>> best_model, best_hyperparams, all_results = optimizer.hyperparam_search(params, train_dataset, test_dataset, metric)
  >>> best_hyperparams['penalty'], best_hyperparams['solver']  # doctest: +SKIP
  ('l2', 'saga')

  Parameters
  ----------
  model_builder: constructor function.
    This parameter must be constructor function which returns an
    object which is an instance of `dc.models.Model`. This function
    must accept two arguments, `model_params` of type `dict` and
    `model_dir`, a string specifying a path to a model directory.
  max_iter: int
    Maximum number of iterations to perform
  """

  def __init__(self, model_builder: Callable[..., Model], max_iter: int):
    super(RandomHyperparamOpt, self).__init__(model_builder=model_builder)
    self.max_iter = max_iter

  def hyperparam_search(
      self,
      params_dict: Dict,
      train_dataset: Dataset,
      valid_dataset: Dataset,
      metric: Metric,
      output_transformers: List[Transformer] = [],
      nb_epoch: int = 10,
      use_max: bool = True,
      logfile: str = 'results.txt',
      logdir: Optional[str] = None,
      **kwargs,
  ) -> Tuple[Model, Dict[str, Any], Dict[str, Any]]:
    """Perform random hyperparams search according to `params_dict`.

    Each key of the `params_dict` is a model_param. The
    values should either be a list of potential values of that hyperparam
    or a callable which can generate random samples.

    Parameters
    ----------
    params_dict: Dict
      Maps each hyperparameter name (string) to either a list of possible
      parameter values or a callable which can generate random samples.
    train_dataset: Dataset
      dataset used for training
    valid_dataset: Dataset
      dataset used for validation (optimization on valid scores)
    metric: Metric
      metric used for evaluation
    output_transformers: list[Transformer]
      Transformers for evaluation. This argument is needed since
      `train_dataset` and `valid_dataset` may have been transformed
      for learning and need the transform to be inverted before
      the metric can be evaluated on a model.
    nb_epoch: int, (default 10)
      Specifies the number of training epochs during each iteration of optimization.
      Not used by all model types.
    use_max: bool, optional
      If True, return the model with the highest score. Else return
      model with the minimum score.
    logdir: str, optional
      The directory in which to store created models. If not set, will
      use a temporary directory.
    logfile: str, optional (default `results.txt`)
      Name of logfile to write results to. If specified, this is must
      be a valid file name. If not specified, results of hyperparameter
      search will be written to `logdir/results.txt`.

    Returns
    -------
    Tuple[`best_model`, `best_hyperparams`, `all_scores`]
      `(best_model, best_hyperparams, all_scores)` where `best_model` is
      an instance of `dc.model.Model`, `best_hyperparams` is a
      dictionary of parameters, and `all_scores` is a dictionary mapping
      string representations of hyperparameter sets to validation
      scores.
    """

    # hyperparam_list should either be an Iterable sequence or a random sampler with rvs method
    for hyperparam in params_dict.values():
      assert isinstance(hyperparam,
                        collections.abc.Iterable) or callable(hyperparam)

    if use_max:
      best_validation_score = -np.inf
    else:
      best_validation_score = np.inf

    best_model = None
    all_scores = {}

    if logdir is not None:
      if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
      log_file = os.path.join(logdir, logfile)

    hyperparameter_combs = RandomHyperparamOpt.generate_random_hyperparam_values(
        params_dict, self.max_iter)

    for ind, model_params in enumerate(hyperparameter_combs):
      logger.info("Fitting model %d/%d" % (ind + 1, self.max_iter))
      logger.info("hyperparameters: %s" % str(model_params))

      hp_str = _convert_hyperparam_dict_to_filename(model_params)

      if logdir is not None:
        model_dir = os.path.join(logdir, hp_str)
        logger.info("model_dir is %s" % model_dir)
        try:
          os.makedirs(model_dir)
        except OSError:
          if not os.path.isdir(model_dir):
            logger.info("Error creating model_dir, using tempfile directory")
            model_dir = tempfile.mkdtemp()
      else:
        model_dir = tempfile.mkdtemp()

      model_params['model_dir'] = model_dir
      model = self.model_builder(**model_params)

      # mypy test throws error, so ignoring it in try
      try:
        model.fit(train_dataset, nb_epoch=nb_epoch)  # type: ignore
      # Not all models have nb_epoch
      except TypeError:
        model.fit(train_dataset)
      try:
        model.save()
      # Some models autosave
      except NotImplementedError:
        pass

      multitask_scores = model.evaluate(valid_dataset, [metric],
                                        output_transformers)
      valid_score = multitask_scores[metric.name]

      # Update best validation score so far
      if (use_max and valid_score >= best_validation_score) or (
          not use_max and valid_score <= best_validation_score):
        best_validation_score = valid_score
        best_hyperparams = model_params
        best_model = model
        all_scores[hp_str] = valid_score

      # if `hyp_str` not in `all_scores`, store it in `all_scores`
      if hp_str not in all_scores:
        all_scores[hp_str] = valid_score

      logger.info("Model %d/%d, Metric %s, Validation set %s: %f" %
                  (ind + 1, nb_epoch, metric.name, ind, valid_score))
      logger.info("\tbest_validation_score so far: %f" % best_validation_score)

    if best_model is None:

      logger.info("No models trained correctly.")

      # arbitrarily return last model trained
      if logdir is not None:
        with open(log_file, 'w+') as f:
          f.write("No model trained correctly. Arbitary models returned")

      best_model, best_hyperparams = model, model_params
      return best_model, best_hyperparams, all_scores

    multitask_scores = best_model.evaluate(train_dataset, [metric],
                                           output_transformers)
    train_score = multitask_scores[metric.name]
    logger.info("Best hyperparameters: %s" % str(best_hyperparams))
    logger.info("best train_score: %f" % train_score)
    logger.info("best validation_score: %f" % best_validation_score)

    if logdir is not None:
      with open(log_file, 'w+') as f:
        f.write("Best Hyperparameters dictionary %s\n" % str(best_hyperparams))
        f.write("Best validation score %f\n" % best_validation_score)
        f.write("Best train_score: %f\n" % train_score)
    return best_model, best_hyperparams, all_scores

  @classmethod
  def generate_random_hyperparam_values(cls, params_dict: Dict,
                                        n: int) -> List[Dict[str, Any]]:
    """Generates `n` random hyperparameter combinations of hyperparameter values

    Parameters
    ----------
    params_dict: Dict
      A dictionary of hyperparameters where parameter which takes discrete
      values are specified as iterables and continuous parameters are of
      type callables.
    n: int
      Number of hyperparameter combinations to generate

    Returns
    -------
    A list of generated hyperparameters

    Example
    -------
    >>> from scipy.stats import uniform
    >>> from deepchem.hyper import RandomHyperparamOpt
    >>> n = 1
    >>> params_dict = {'a': [1, 2, 3], 'b': [5, 7, 8], 'c': uniform(10, 5).rvs}
    >>> RandomHyperparamOpt.generate_random_hyperparam_values(params_dict, n)  # doctest: +SKIP
    [{'a': 3, 'b': 7, 'c': 10.619700740985433}]
    """
    hyperparam_keys, hyperparam_values = [], []
    for key, values in params_dict.items():
      if callable(values):
        # If callable, sample it for a maximum n times
        values = [values() for i in range(n)]
      hyperparam_keys.append(key)
      hyperparam_values.append(values)

    hyperparam_combs = []
    for iterable_hyperparam_comb in itertools.product(*hyperparam_values):
      hyperparam_comb = list(iterable_hyperparam_comb)
      hyperparam_combs.append(hyperparam_comb)

    indices = np.random.permutation(len(hyperparam_combs))[:n]
    params_subset = []
    for index in indices:
      param = {}
      for key, hyperparam_value in zip(hyperparam_keys,
                                       hyperparam_combs[index]):
        param[key] = hyperparam_value
      params_subset.append(param)
    return params_subset
