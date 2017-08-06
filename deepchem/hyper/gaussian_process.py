"""
Contains basic hyperparameter optimizations.
"""
import numpy as np
import os
import itertools
import tempfile
import shutil
import collections
from functools import reduce
from operator import mul
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.save import log
from deepchem.hyper import HyperparamOpt
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
                            'seed', 'nb_epoch', 'penalty_type',
                            'dropouts', 'bypass_dropouts',
                            'n_pair_feat'
                        ],
                        logdir=None):
    assert len(metric) == 1, 'Only use one metric'
    hyper_parameters = params_dict
    hp_list = hyper_parameters.keys()
    for hp in hp_invalid_list:
      if hp in hp_list:
        hp_list.remove(hp)

    hp_list_class = [hyper_parameters[hp].__class__ for hp in hp_list]
    assert set(hp_list_class) <= set([list, int, float])
    # Float or int hyper parameters(ex. batch_size, learning_rate)
    hp_list1 = [
        hp_list[i] for i in range(len(hp_list)) if not hp_list_class[i] is list
    ]
    # List of float or int hyper parameters(ex. layer_sizes)
    hp_list2 = [(hp_list[i], len(hyper_parameters[hp_list[i]]))
                for i in range(len(hp_list)) if hp_list_class[i] is list]

    # Number of parameters
    n_param = len(hp_list1) + sum([hp[1] for hp in hp_list2])
    # Range of optimization
    param_range = []
    for hp in hp_list1:
      if hyper_parameters[hp].__class__ is int:
        param_range.append((('int'), [
            hyper_parameters[hp] / search_range,
            hyper_parameters[hp] * search_range
        ]))
      else:
        param_range.append((('cont'), [
            hyper_parameters[hp] / search_range,
            hyper_parameters[hp] * search_range
        ]))
    for hp in hp_list2:
      if hyper_parameters[hp[0]][0].__class__ is int:
        param_range.extend([(('int'), [
            hyper_parameters[hp[0]][i] / search_range,
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
      args = locals()
      # Input hyper parameters
      i = 0
      for hp in hp_list1:
        hyper_parameters[hp] = float(args[param_name[i]])
        if param_range[i][0] == 'int':
          hyper_parameters[hp] = int(hyper_parameters[hp])
        i = i + 1
      for hp in hp_list2:
        hyper_parameters[hp[0]] = [
            float(args[param_name[j]]) for j in range(i, i + hp[1])
        ]
        if param_range[i][0] == 'int':
          hyper_parameters[hp[0]] = map(int, hyper_parameters[hp[0]])
        i = i + hp[1]

      print(hyper_parameters)
      # Run benchmark
      if isinstance(self.model_class, str) or isinstance(self.model_class, unicode):
        try:
          train_scores, valid_scores, _ = benchmark_classification(
              train_dataset, valid_dataset, None, ['task_placeholder']*n_tasks,
              output_transformers, n_features, metric, model, 
              hyper_parameters=hyper_parameters)
        except AssertionError:
          train_scores, valid_scores, _ = benchmark_regression(
              train_dataset, valid_dataset, None, ['task_placeholder']*n_tasks,
              output_transformers, n_features, metric, model, 
              hyper_parameters=hyper_parameters)
        return valid_scores[model][metric[0].name]
      else:
        model_dir = tempfile.mkdtemp()
        model = self.model_class(hyper_parameters, model_dir)
        model.fit(train_dataset, **hyper_parameters)
        model.save()
        evaluator = Evaluator(model, valid_dataset, output_transformers)
        multitask_scores = evaluator.compute_model_performance([metric])
        return multitask_scores[metric.name]

    cov = pyGPGO_covfunc.matern32()
    gp = pyGPGO_surrogates_GaussianProcess.GaussianProcess(cov)
    acq = pyGPGO_acquisition.Acquisition(mode='ExpectedImprovement')
    gpgo = pyGPGO_GPGO.GPGO(gp, acq, f, param)
    gpgo.run(max_iter=max_iter)

    hp_opt, valid_performance_opt = gpgo.getResult()

    # Readout best hyper parameters
    i = 0
    for hp in hp_list1:
      hyper_parameters[hp] = float(hp_opt[param_name[i]])
      if param_range[i][0] == 'int':
        hyper_parameters[hp] = int(hyper_parameters[hp])
      i = i + 1
    for hp in hp_list2:
      hyper_parameters[hp[0]] = [
          float(hp_opt[param_name[j]]) for j in range(i, i + hp[1])
      ]
      if param_range[i][0] == 'int':
        hyper_parameters[hp[0]] = map(int, hyper_parameters[hp[0]])
      i = i + hp[1]

    return hyper_parameters, valid_performance_opt
