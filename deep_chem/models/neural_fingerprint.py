"""
Code for training neural-fingerprint models in deep_chem framework.
"""

import autograd.numpy as np
import autograd.numpy.random as npr
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint.util import rmse
from autograd import grad

def fit_neural_fingerprints(train_data, task_types, **training_params):
  """Fit neural fingerprint model."""
  # TODO(rbharath): Code smell here. Looks just like fit_singletask_mlp. Is it
  # worth factoring out core logic here?
  models = {}
  for index, target in enumerate(sorted(train_data.keys())):
    print "Training model %d" % index
    print "Target %s" % target
    (ids, X_train, y_train, W_train) = train_data[target] 
    if task_types[target] != "regression":
      raise ValueError("Only regression supported for neural fingerprints.")
    models[target] = train_neural_fingerprint(X_train, y_train, training_params)
  return models

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets,
             train_params, seed=0, validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print "Total number of weights in the network:", num_weights
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
      if iter % 10 == 0:
        print "max of weights", np.max(np.abs(weights))
        train_preds = undo_norm(pred_fun(weights, train_smiles))
        cur_loss = loss_fun(weights, train_smiles, train_targets)
        training_curve.append(cur_loss)
        print "Iteration", iter, "loss", cur_loss, "train RMSE", rmse(train_preds, train_raw_targets),
        if validation_smiles is not None:
          validation_preds = undo_norm(pred_fun(weights, validation_smiles))
          print "Validation RMSE", iter, ":", rmse(validation_preds, validation_raw_targets),

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'],
                           step_size=train_params['step_size'],
                           b1=train_params['b1'], b2=train_params['b2'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve

# TODO(rbharath): X_train needs to be made a ndarray of smiles string.
# TODO(rbharath): training_params needs to be hooked up with modeler.
def train_neural_fingerprint(X_train, y_train, training_params):
  """Trains a neural fingerprint model."""
  model_params = dict(fp_length = 512,   # Usually neural fps need far fewer
                                         # dimensions than morgan.
                      fp_depth = 4,      # The depth of the network equals the
                                         # fingerprint radius.
                      conv_width = 20,   # Only the neural fps need this parameter.
                      h1_size = 100,     # Size of hidden layer of network on top of fps.
                      L2_reg = -2)
  train_params = dict(num_iters = 10,
                      batch_size = 100,
                      init_scale = np.exp(-4),
                      step_size = np.exp(-6),
                      b1 = np.exp(-3),   # Parameter for Adam optimizer.
                      b2 = np.exp(-2))   # Parameter for Adam optimizer.
  # Define the architecture of the network that sits on top of the fingerprints.
  vanilla_net_params = dict(
      layer_sizes = [model_params['fp_length'], model_params['h1_size']],  # One hidden layer.
      normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)

  conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
  conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                      'fp_length' : model_params['fp_length'], 'normalize' : 1}
  print "np.shape(X_train)"
  print np.shape(X_train)
  print "X_train[:10]"
  print X_train[:10]
  loss_fun, pred_fun, conv_parser =  build_conv_deep_net(
      conv_arch_params, vanilla_net_params, model_params['L2_reg'])
  num_weights = len(conv_parser)
  predict_func, trained_weights, conv_training_curve = train_nn(
      pred_fun, loss_fun, num_weights, X_train, y_train, train_params)
  return (predict_func, train_weights, conv_training_curve)
