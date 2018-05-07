from nose.tools import timed
@timed(180)
def test_notebook():
  
  # coding: utf-8
  
  # # Quantum Machinery with gdb1k
  
  # In[ ]:
  
  
  get_ipython().run_line_magic('load_ext', 'autoreload')
  get_ipython().run_line_magic('autoreload', '2')
  get_ipython().run_line_magic('pdb', 'off')
  __author__ = "Joseph Gomes and Bharath Ramsundar"
  __copyright__ = "Copyright 2016, Stanford University"
  __license__ = "LGPL"
  
  import os
  import unittest
  
  import numpy as np
  import deepchem as dc
  import numpy.random
  from deepchem.utils.evaluate import Evaluator
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.kernel_ridge import KernelRidge
  
  
  # Setting up model variables
  
  # In[ ]:
  
  
  featurizer = dc.feat.CoulombMatrixEig(23, remove_hydrogens=False)
  tasks = ["atomization_energy"]
  dataset_file = "../../datasets/gdb1k.sdf"
  smiles_field = "smiles"
  mol_field = "mol"
  
  
  # Load featurized data
  
  # In[ ]:
  
  
  loader = dc.data.SDFLoader(
  tasks=["atomization_energy"], smiles_field="smiles",
  featurizer=featurizer,
  mol_field="mol")
  dataset = loader.featurize(dataset_file)
  
  
  # Perform Train, Validation, and Testing Split
  
  # In[ ]:
  
  
  random_splitter = dc.splits.RandomSplitter()
  train_dataset, valid_dataset, test_dataset = random_splitter.train_valid_test_split(dataset)
  
  
  # Transforming datasets
  
  # In[ ]:
  
  
  transformers = [
  dc.trans.NormalizationTransformer(transform_X=True, dataset=train_dataset),
  dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
  
  for dataset in [train_dataset, valid_dataset, test_dataset]:
  for transformer in transformers:
  dataset = transformer.transform(dataset)
  
  
  # Fit Random Forest with hyperparameter search
  
  # In[ ]:
  
  
  def rf_model_builder(model_params, model_dir):
  sklearn_model = RandomForestRegressor(**model_params)
  return dc.models.SklearnModel(sklearn_model, model_dir)
  params_dict = {
  "n_estimators": [10, 100],
  "max_features": ["auto", "sqrt", "log2", None],
  }
  
  metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
  optimizer = dc.hyper.HyperparamOpt(rf_model_builder)
  best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
  params_dict, train_dataset, valid_dataset, transformers,
  metric=metric)
  
  
  # In[ ]:
  
  
  def krr_model_builder(model_params, model_dir):
  sklearn_model = KernelRidge(**model_params)
  return dc.models.SklearnModel(sklearn_model, model_dir)
  
  params_dict = {
  "kernel": ["laplacian"],
  "alpha": [0.0001],
  "gamma": [0.0001]
  }
  
  metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
  optimizer = dc.hyper.HyperparamOpt(krr_model_builder)
  best_krr, best_krr_hyperparams, all_krr_results = optimizer.hyperparam_search(
  params_dict, train_dataset, valid_dataset, transformers,
  metric=metric)
  
