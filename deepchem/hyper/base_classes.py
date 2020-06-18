
def compute_parameter_search_space(params_dict, search_range):
  """Convenience Function to compute parameter search space.

  Parameters
  ----------
  params_dict: dict
    Dictionary mapping strings to Ints/Floats/Lists. For those
    parameters in which int/float is specified, an explicit list of
    parameters is computed with `search_range`. Parameters in `hp_invalid_list`
  search_range: int(float) (default 4)
    For int/float values in `params_dict`, computes optimization range
    on `[initial values / search_range, initial values *
    search_range]`

  Returns
  -------
  expanded_params: dict
    Expanded dictionary of parameters where all int/float values in
    `params_dict` are expanded out into explicit search ranges.
  """
  hyper_parameters = params_dict
  hp_list = list(hyper_parameters.keys())

  for hp in hp_invalid_list:
    if hp in hp_list:
      hp_list.remove(hp)

  hp_list_class = [hyper_parameters[hp].__class__ for hp in hp_list]
  # Check the type is correct
  if not (set(hp_list_class) <= set([list, int, float])):
    raise ValueError("params_dict must contain values that are lists/ints/floats.")

  # Float or int hyper parameters(ex. batch_size, learning_rate)
  hp_list_single = [
      hp_list[i] for i in range(len(hp_list)) if not hp_list_class[i] is list
  ]

  # List of float or int hyper parameters(ex. layer_sizes)
  hp_list_multiple = [(hp_list[i], len(hyper_parameters[hp_list[i]]))
                      for i in range(len(hp_list))
                      if hp_list_class[i] is list]

  # Range of optimization
  param_range = []
  for hp in hp_list_single:
    if hyper_parameters[hp].__class__ is int:
      param_range.append((('int'), [
          hyper_parameters[hp] // search_range,
          hyper_parameters[hp] * search_range
      ]))
    else:
      param_range.append((('cont'), [
          hyper_parameters[hp] / search_range,
          hyper_parameters[hp] * search_range
      ]))
  for hp in hp_list_multiple:
    if hyper_parameters[hp[0]][0].__class__ is int:
      param_range.extend([(('int'), [
          hyper_parameters[hp[0]][i] // search_range,
          hyper_parameters[hp[0]][i] * search_range
      ]) for i in range(hp[1])])
    else:
      param_range.extend([(('cont'), [
          hyper_parameters[hp[0]][i] / search_range,
          hyper_parameters[hp[0]][i] * search_range
      ]) for i in range(hp[1])])

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

  Objects of this class maintains a list of constants,
  `hp_invalid_list` that contains a list of model parameters which
  cannot be optimized over This list is used to catch user errors. You
  can customize this list in the constructor.
  """

  def __init__(self, model_class, hp_invalid_list=['seed', 'nb_epoch', 'penalty_type', 'dropouts', 'bypass_dropouts', 'n_pair_feat', 'fit_transformers', 'min_child_weight', 'max_delta_step', 'subsample', 'colsample_bylevel', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'scale_pos_weight', 'base_score']):
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
    hp_invalid_list: list, (default `['seed', 'nb_epoch', 'penalty_type', 'dropouts', 'bypass_dropouts', 'n_pair_feat', 'fit_transformers', 'min_child_weight', 'max_delta_step', 'subsample', 'colsample_bylevel', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'scale_pos_weight', 'base_score']`)
    """
    if self.__class__.__name__ == "HyperparamOpt":
      raise ValueError(
          "HyperparamOpt is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
      )
    self.model_class = model_class
    self.hp_invalid_list = hp_invalid_list

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
      Dictionary mapping strings to Ints/Floats/Lists. For those
      parameters in which int/float is specified, an explicit list of
      parameters is computed with `search_range`.
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
