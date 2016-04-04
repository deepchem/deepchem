"""
Contains basic hyperparameter optimizations.
"""
import numpy as np
import itertools
import tempfile
import shutil
import collections
from operator import mul
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.save import log

class HyperparamOpt(object):
  """
  Provides simple hyperparameter search capabilities.
  """

  def __init__(self, model_class, task_types, fit_transformers=None, verbosity=None):
    self.model_class = model_class
    self.task_types = task_types
    self.fit_transformers = fit_transformers
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity

  # TODO(rbharath): This function is complicated and monolithic. Is there a nice
  # way to refactor this?
  def hyperparam_search(self, params_dict, train_dataset, valid_dataset,
                        output_transformers, metric, use_max=True,
                        logdir=None):
    """Perform hyperparams search according to params_dict.
    
    Each key to hyperparams_dict is a model_param. The values should be a list
    of potential values for that hyperparam. 
    """
    hyperparams = params_dict.keys()
    hyperparam_vals = params_dict.values() 
    for hyperparam_list in params_dict.itervalues():
      assert isinstance(hyperparam_list, collections.Iterable)

    number_combinations = reduce(mul, [len(vals) for vals in hyperparam_vals])

    valid_csv_out = tempfile.NamedTemporaryFile()
    valid_stats_out = tempfile.NamedTemporaryFile()
    if use_max:
      best_validation_score = -np.inf
    else:
      best_validation_score = np.inf
    best_hyperparams = None
    best_model, best_model_dir = None, None
    all_scores = {}
    for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparam_vals)):
      model_params = {}
      for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
        model_params[hyperparam] = hyperparam_val

      if logdir is not None:
        model_dir = logdir
      else:
        model_dir = tempfile.mkdtemp()
      if logdir is not None:
        #TODO(JG) Fit transformers for TF models
        model = self.model_class(self.task_types, model_params, model_dir,
                                 verbosity=self.verbosity)
      else:
        if self.fit_transformers:
          model = self.model_class(self.task_types, model_params,
                                   fit_transformers=self.fit_transformers,
                                   verbosity=self.verbosity)
        else:
          model = self.model_class(self.task_types, model_params,
                                   verbosity=self.verbosity)
        
      model.fit(train_dataset)
      model.save(model_dir)
    
      evaluator = Evaluator(model, valid_dataset, output_transformers)
      df, scores_df, multitask_scores = evaluator.compute_model_performance(
          [metric], valid_csv_out, valid_stats_out)
      if not metric.is_multitask:
        valid_score = scores_df.iloc[0][metric.name]
      else:
        valid_score = multitask_scores[metric.name]
      all_scores[hyperparameter_tuple] = valid_score
    
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
  
      log("Model %d/%d, Metric %s, Validation set %s: %f" %
          (ind, number_combinations, metric.name, ind, valid_score),
          self.verbosity)
      log("\tbest_validation_score so far: %f" % best_validation_score,
          self.verbosity)

    if best_model is None:
      log("No models trained correctly.", self.verbosity)
      return best_model, best_hyperparams, all_scores
    train_csv_out = tempfile.NamedTemporaryFile()
    train_stats_out = tempfile.NamedTemporaryFile()
    train_evaluator = Evaluator(best_model, train_dataset, output_transformers)
    train_df, train_score, multitask_scores = train_evaluator.compute_model_performance(
        [metric], train_csv_out, train_stats_out)
    if not metric.is_multitask:
      train_score = train_score.iloc[0][metric.name]
    else:
      train_score = multitask_scores[metric.name]
    log("Best hyperparameters: %s" % str(best_hyperparams),
        self.verbosity)
    log("train_score: %f" % train_score, self.verbosity)
    log("validation_score: %f" % best_validation_score, self.verbosity)
    return best_model, best_hyperparams, all_scores
