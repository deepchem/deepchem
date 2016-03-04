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

class HyperparamOpt(object):
  """
  Provides simple hyperparameter search capabilities.
  """

  def __init__(self, model_class, task_types):
    self.model_class = model_class
    self.task_types = task_types

  def hyperparam_search(self, params_dict, train_dataset, valid_dataset,
                        output_transformers, metric, use_max=True, verbosity=None):
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
    if use_max == True:
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

      model_dir = tempfile.mkdtemp()
      model = self.model_class(self.task_types, model_params, verbosity=verbosity)
      model.fit(train_dataset)
      model.save(model_dir)
    
      evaluator = Evaluator(model, valid_dataset, output_transformers)
      df, score = evaluator.compute_model_performance(
          valid_csv_out, valid_stats_out)
      valid_score = score.iloc[0][metric]
      all_scores[hyperparameter_tuple] = valid_score
      if (use_max == True and valid_score > best_validation_score) or (
            use_max == False and valid_score < best_validation_score):
        best_validation_score = valid_score
        best_hyperparams = hyperparameter_tuple
        if best_model_dir is not None:
            shutil.rmtree(best_model_dir)
        best_model_dir = model_dir
        best_model = model
      else:
        shutil.rmtree(model_dir)

      print("Model %d/%d, Metric %s, Validation set %s: %f" %
            (ind, number_combinations, metric, ind, valid_score))
      print("\tbest_validation_score so  far: %f" % best_validation_score)


    train_csv_out = tempfile.NamedTemporaryFile()
    train_stats_out = tempfile.NamedTemporaryFile()
    train_evaluator = Evaluator(best_model, train_dataset, output_transformers)
    train_df, train_score = train_evaluator.compute_model_performance(
        train_csv_out, train_stats_out)
    train_score = train_score.iloc[0][metric]
    print("Best hyperparameters: %s" % str(zip(hyperparams, best_hyperparams)))
    print("train_score: %f" % train_score)
    print("validation_score: %f" % best_validation_score)
    return best_model, best_hyperparams, all_scores
