"""
Contains class for gaussian process hyperparameter optimizations.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tempfile
from deepchem.hyper.grid_search import HyperparamOpt
from deepchem.utils.evaluate import Evaluator
from deepchem.molnet.run_benchmark_models import benchmark_classification, benchmark_regression


class GaussianProcessHyperparamOpt(HyperparamOpt):
  """
  Gaussian Process Global Optimization(GPGO)
  """

  def hyperparam_search(self,
                        params_dict,
                        train_dataset,
                        valid_dataset,
                        output_transformers,
                        metric,
                        n_features=1024,
                        n_tasks=1,
                        max_iter=20,
                        search_range=4,
                        hp_invalid_list=[
                            'seed', 'nb_epoch', 'penalty_type', 'dropouts',
                            'bypass_dropouts', 'n_pair_feat'
                        ],
                        logdir=None):
    """Perform hyperparams search using a gaussian process assumption

    params_dict include single-valued parameters being optimized,
    which should only contain int, float and list of int(float)

    parameters with names in hp_invalid_list will not be changed.

    For Molnet models, self.model_class is model name in string,
    params_dict = dc.molnet.preset_hyper_parameters.hps[self.model_class]

    Parameters
    ----------
    params_dict: dict
      dict including parameters and their initial values
      parameters not suitable for optimization can be added to hp_invalid_list
    train_dataset: dc.data.Dataset struct
      dataset used for training
    valid_dataset: dc.data.Dataset struct
      dataset used for validation(optimization on valid scores)
    output_transformers: list of dc.trans.Transformer
      transformers for evaluation
    metric: list of dc.metrics.Metric
      metric used for evaluation
    n_features: int
      number of input features
    n_tasks: int
      number of tasks
    max_iter: int
      number of optimization trials
    search_range: int(float)
      optimization on [initial values / search_range,
                       initial values * search_range]
    hp_invalid_list: list
      names of parameters that should not be optimized

    Returns
    -------
    hyper_parameters: dict
      params_dict with all optimized values
    valid_performance_opt: float
      best performance on valid dataset

    """

    assert len(metric) == 1, 'Only use one metric'
    hyper_parameters = params_dict
    hp_list = hyper_parameters.keys()
    for hp in hp_invalid_list:
      if hp in hp_list:
        hp_list.remove(hp)

    hp_list_class = [hyper_parameters[hp].__class__ for hp in hp_list]
    assert set(hp_list_class) <= set([list, int, float])
    # Float or int hyper parameters(ex. batch_size, learning_rate)
    hp_list_single = [
        hp_list[i] for i in range(len(hp_list)) if not hp_list_class[i] is list
    ]
    # List of float or int hyper parameters(ex. layer_sizes)
    hp_list_multiple = [(hp_list[i], len(hyper_parameters[hp_list[i]]))
                        for i in range(len(hp_list))
                        if hp_list_class[i] is list]

    # Number of parameters
    n_param = len(hp_list_single + sum([hp[1] for hp in hp_list_multiple]))
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

    # Dummy names
    param_name = ['l' + format(i, '02d') for i in range(20)]
    param = dict(zip(param_name[:n_param], param_range))

    def f(l00=0,
          l01=0,
          l02=0,
          l03=0,
          l04=0,
          l05=0,
          l06=0,
          l07=0,
          l08=0,
          l09=0,
          l10=0,
          l11=0,
          l12=0,
          l13=0,
          l14=0,
          l15=0,
          l16=0,
          l17=0,
          l18=0,
          l19=0):
      """ Optimizing function
      Take in hyper parameter values and return valid set performances

      Parameters
      ----------
      l00~l19: int or float
        placeholders for hyperparameters being optimized,
        hyper_parameters dict is rebuilt based on input values of placeholders

      Returns:
      --------
      valid_scores: float
        valid set performances
      """
      args = locals()
      # Input hyper parameters
      i = 0
      for hp in hp_list_single:
        hyper_parameters[hp] = float(args[param_name[i]])
        if param_range[i][0] == 'int':
          hyper_parameters[hp] = int(hyper_parameters[hp])
        i = i + 1
      for hp in hp_list_multiple:
        hyper_parameters[hp[0]] = [
            float(args[param_name[j]]) for j in range(i, i + hp[1])
        ]
        if param_range[i][0] == 'int':
          hyper_parameters[hp[0]] = map(int, hyper_parameters[hp[0]])
        i = i + hp[1]

      print(hyper_parameters)
      # Run benchmark
      if isinstance(self.model_class, str) or isinstance(
          self.model_class, unicode):
        try:
          train_scores, valid_scores, _ = benchmark_classification(
              train_dataset,
              valid_dataset,
              valid_dataset, ['task_placeholder'] * n_tasks,
              output_transformers,
              n_features,
              metric,
              self.model_class,
              hyper_parameters=hyper_parameters)
        except AssertionError:
          train_scores, valid_scores, _ = benchmark_regression(
              train_dataset,
              valid_dataset,
              valid_dataset, ['task_placeholder'] * n_tasks,
              output_transformers,
              n_features,
              metric,
              self.model_class,
              hyper_parameters=hyper_parameters)
        return valid_scores[self.model_class][metric[0].name]
      else:
        model_dir = tempfile.mkdtemp()
        model = self.model_class(hyper_parameters, model_dir)
        model.fit(train_dataset, **hyper_parameters)
        model.save()
        evaluator = Evaluator(model, valid_dataset, output_transformers)
        multitask_scores = evaluator.compute_model_performance([metric])
        return multitask_scores[metric.name]

    import pyGPGO
    cov = pyGPGO.covfunc.matern32()
    gp = pyGPGO.surrogates.GaussianProcess.GaussianProcess(cov)
    acq = pyGPGO.acquisition.Acquisition(mode='ExpectedImprovement')
    gpgo = pyGPGO.GPGO.GPGO(gp, acq, f, param)
    gpgo.run(max_iter=max_iter)

    hp_opt, valid_performance_opt = gpgo.getResult()

    # Readout best hyper parameters
    i = 0
    for hp in hp_list_single:
      hyper_parameters[hp] = float(hp_opt[param_name[i]])
      if param_range[i][0] == 'int':
        hyper_parameters[hp] = int(hyper_parameters[hp])
      i = i + 1
    for hp in hp_list_multiple:
      hyper_parameters[hp[0]] = [
          float(hp_opt[param_name[j]]) for j in range(i, i + hp[1])
      ]
      if param_range[i][0] == 'int':
        hyper_parameters[hp[0]] = map(int, hyper_parameters[hp[0]])
      i = i + hp[1]

    return hyper_parameters, valid_performance_opt
