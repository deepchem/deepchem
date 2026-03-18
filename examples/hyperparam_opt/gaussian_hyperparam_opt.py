import numpy as np
import deepchem as dc
np.random.seed(123)

# Load delaney dataset
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
    featurizer="GraphConv")
train, valid, test = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
optimizer = dc.hyper.GaussianProcessHyperparamOpt(
    lambda **p: dc.models.GraphConvModel(n_tasks=len(delaney_tasks), mode="regression", **p))

params_dict = {"dropout": 0.5}
best_model, best_params, all_results = optimizer.hyperparam_search(
    params_dict, train, valid, metric, transformers, max_iter=1, search_range=2)

valid_score = best_model.evaluate(valid, [metric], transformers)
print("valid_score")
print(valid_score)
