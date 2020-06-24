import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc
import sklearn

# Load delaney dataset
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney()
train, valid, test = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)


def rf_model_builder(**model_params):
  rf_params = {k: v for (k, v) in model_params.items() if k != 'model_dir'}
  model_dir = model_params['model_dir']
  sklearn_model = sklearn.ensemble.RandomForestRegressor(**rf_params)
  return dc.models.SklearnModel(sklearn_model, model_dir)


params_dict = {"n_estimators": 30}
optimizer = dc.hyper.GaussianProcessHyperparamOpt(rf_model_builder)
best_model, best_params, all_results = optimizer.hyperparam_search(
    params_dict, train, valid, transformers, metric)
