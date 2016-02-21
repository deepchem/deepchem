"""
Contains basic hyperparameter optimizations.
"""
import tempfile
import shutil

model_params = {"activation": "relu",
                "momentum": .9,
                "batch_size": 50,
                "init": "glorot_uniform",
                "data_shape": train_dataset.get_data_shape()}
lr_list = np.power(10., np.random.uniform(-5, -2, size=5))
decay_list = np.power(10, np.random.uniform(-6, -4, size=5))
nb_hidden_list = [500, 1000]
nb_epoch_list = [40]
nesterov_list = [False]
dropout_list = [0, .5]
nb_layers_list = [1]
batchnorm_list = [False]

#hyperparameters = [lr_list, decay_list, nb_layers_list,
#                   nb_hidden_list, nb_epoch_list, nesterov_list,
#                   dropout_list, batchnorm_list]
#    model_params["learning_rate"] = lr
#    model_params["decay"] = decay
#    model_params["nb_layers"] = nb_layers
#    model_params["nb_hidden"] = nb_hidden
#    model_params["nb_epoch"] = nb_epoch
#    model_params["nesterov"] = nesterov
#    model_params["dropout"] = dropout
#    model_params["batchnorm"] = batchnorm

class HyperparameterOpt(object):
  """
  Provides simple hyperparameter search capabilities.
  """

  def __init__(model_class, task_types):
    self.model_class = model_class
    self.task_types = task_types

  def hyperparam_search(params_dict, train_dataset, validation_dataset,
                        metric="r2_score"):
    """Perform hyperparams search according to params_dict.
    
    Each key to hyperparams_dict is a model_param. The values should be a list
    of potential values for that hyperparam. 
    """
    hyperparams = params_dict.keys()
    hyperparam_vals = [params_dict[hyperparam] for hyperparam in hyperparams]
    for hyperparam_list in hyperparams:
      assert isinstance(hyperparam_list, list)

    best_validation_score = -np.inf
    best_hyperparams = None
    best_model, best_model_dir = None, None
    all_scores = {}
    for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparameters)):
      for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
        model_params[hyperparam] = hyperparam_val

      model_dir = tempfile.mkdtemp()
      model = self.model_class(task_types, model_params, verbosity=None)
      model.fit(train_dataset)
      model.save(model_dir)
    
      evaluator = Evaluator(model, valid_dataset, output_transformers)
      df, score = evaluator.compute_model_performance(
          valid_csv_out, valid_stats_out)
      valid_score = score.iloc[0][metric]
      print("Model %d, Validation set %s: %f" % (metric, ind, valid_score))
      all_scores[hyperparameter_tuple] = valid_score
    
      if valid_score > best_validation_score:
        best_validation_score = valid_score
        best_hyperparams = hyperparameter_tuple
        if best_model_dir is not None:
            shutil.rmtree(best_model_dir)
        best_model_dir = model_dir
        best_model = model
      else:
        shutil.rmtree(model_dir)

    print("Best hyperparameters: %s" % str(best_hyperparams))
    print("best_validation_score: %f" % best_validation_score)
    return best_model, best_hyperparams, all_scores
