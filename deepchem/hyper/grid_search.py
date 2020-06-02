"""
Contains basic hyperparameter optimizations.
"""
import numpy as np
import os
import itertools
import tempfile
import shutil
import collections
import logging
from functools import reduce
from operator import mul
from deepchem.utils.evaluate import Evaluator
from deepchem.hyper.base_classes import HyperparamOpt


class GridHyperparamOpt(HyperparamOpt):
  """
  Provides simple grid hyperparameter search capabilities.

  This class performs a grid hyperparameter search over the specified
  hyperparameter space. This implementation is simple and simply does
  a direct iteration over all possible hyperparameters and doesn't use
  parallelization to speed up the search.
  """

  def hyperparam_search(self,
                        params_dict,
                        train_dataset,
                        valid_dataset,
                        output_transformers,
                        metric,
                        use_max=True,
                        logdir=None):
    """Perform hyperparams search according to params_dict.

    Each key to hyperparams_dict is a model_param. The values should
    be a list of potential values for that hyperparam.

    Parameters
    ----------
    params_dict: dict
      dict including parameters and their initial values.
    train_dataset: `dc.data.Dataset`
      dataset used for training
    valid_dataset: `dc.data.Dataset`
      dataset used for validation(optimization on valid scores)
    output_transformers: list of dc.trans.Transformer
      transformers for evaluation
    metric: dc.metrics.Metric
      metric used for evaluation
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
    hyperparams = params_dict.keys()
    hyperparam_vals = params_dict.values()
    for hyperparam_list in params_dict.values():
      assert isinstance(hyperparam_list, collections.Iterable)

    number_combinations = reduce(mul, [len(vals) for vals in hyperparam_vals])

    if use_max:
      best_validation_score = -np.inf
    else:
      best_validation_score = np.inf
    best_hyperparams = None
    best_model, best_model_dir = None, None
    all_scores = {}
    for ind, hyperparameter_tuple in enumerate(
        itertools.product(*hyperparam_vals)):
      model_params = {}
      logger.info("Fitting model %d/%d" % (ind + 1, number_combinations))
      for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
        model_params[hyperparam] = hyperparam_val
      logger.info("hyperparameters: %s" % str(model_params))

      if logdir is not None:
        model_dir = os.path.join(logdir, str(ind))
        logger.info("model_dir is %s" % model_dir)
        try:
          os.makedirs(model_dir)
        except OSError:
          if not os.path.isdir(model_dir):
            logger.info("Error creating model_dir, using tempfile directory")
            model_dir = tempfile.mkdtemp()
      else:
        model_dir = tempfile.mkdtemp()

      model = self.model_class(model_params, model_dir)
      model.fit(train_dataset)

      evaluator = Evaluator(model, valid_dataset, output_transformers)
      multitask_scores = evaluator.compute_model_performance([metric])
      valid_score = multitask_scores[metric.name]
      all_scores[str(hyperparameter_tuple)] = valid_score

      if (use_max and valid_score >= best_validation_score) or (
          not use_max and valid_score <= best_validation_score):
        best_validation_score = valid_score
        best_hyperparams = hyperparameter_tuple
        if best_model_dir is not None:
          shutil.rmtree(best_model_dir)
        best_model_dir = model_dir
        best_model = model
      else:
        shutil.rmtree(model_dir)

      logger.info("Model %d/%d, Metric %s, Validation set %s: %f" %
                  (ind + 1, number_combinations, metric.name, ind, valid_score))
      logger.info("\tbest_validation_score so far: %f" % best_validation_score)
    if best_model is None:
      logger.info("No models trained correctly.")
      # arbitrarily return last model
      best_model, best_hyperparams = model, hyperparameter_tuple
      return best_model, best_hyperparams, all_scores
    train_evaluator = Evaluator(best_model, train_dataset, output_transformers)
    multitask_scores = train_evaluator.compute_model_performance([metric])
    train_score = multitask_scores[metric.name]
    logger.info("Best hyperparameters: %s" % str(best_hyperparams))
    logger.info("train_score: %f" % train_score)
    logger.info("validation_score: %f" % best_validation_score)
    return best_model, best_hyperparams, all_scores
