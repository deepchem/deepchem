import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc

# Load delaney dataset
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney()
train, valid, test= delaney_datasets

# Fit models
regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)


# TODO(rbharath): I don't like this awkward string/class divide. Maybe clean up?
optimizer = dc.hyper.GaussianProcessHyperparamOpt('tf_regression')
best_hyper_params, best_performance = optimizer.hyperparam_search(
  dc.molnet.preset_hyper_parameters.hps['tf_regression'],
  train,
  valid,
  transformers,
  [regression_metric]
)
