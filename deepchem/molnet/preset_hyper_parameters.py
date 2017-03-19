#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:07:10 2017

@author: zqwu
"""
import deepchem

hps = {}
hps['tf'] = {
    'layer_sizes': [1500],
    'weight_init_stddevs': [0.02],
    'bias_init_consts': [1.],
    'dropouts': [0.5],
    'penalty': 0.1,
    'penalty_type': 'l2',
    'batch_size': 50,
    'nb_epoch': 10,
    'learning_rate': 0.001
}

hps['tf_robust'] = {
    'layer_sizes': [1500],
    'weight_init_stddevs': [0.02],
    'bias_init_consts': [1.],
    'dropouts': [0.5],
    'bypass_layer_sizes': [200],
    'bypass_weight_init_stddevs': [0.02],
    'bypass_bias_init_consts': [1.],
    'bypass_dropouts': [0.5],
    'penalty': 0.1,
    'penalty_type': 'l2',
    'batch_size': 50,
    'nb_epoch': 10,
    'learning_rate': 0.0005
}
hps['logreg'] = {
    'penalty': 0.1,
    'penalty_type': 'l2',
    'batch_size': 50,
    'nb_epoch': 10,
    'learning_rate': 0.005
}
hps['irv'] = {
    'penalty': 0.,
    'penalty_type': 'l2',
    'batch_size': 50,
    'nb_epoch': 10,
    'learning_rate': 0.001,
    'n_K': 10
}
hps['graphconv'] = {
    'batch_size': 50,
    'nb_epoch': 15,
    'learning_rate': 0.0005,
    'n_filters': 64,
    'n_fully_connected_nodes': 128,
    'seed': 123
}
hps['rf'] = {'n_estimators': 500}
hps['tf_regression'] = {
    'layer_sizes': [1000, 1000],
    'weight_init_stddevs': [0.02, 0.02],
    'bias_init_consts': [1., 1.],
    'dropouts': [0.25, 0.25],
    'penalty': 0.0005,
    'penalty_type': 'l2',
    'batch_size': 128,
    'nb_epoch': 50,
    'learning_rate': 0.0008
}
hps['tf_regression_ft'] = {
    'layer_sizes': [1000, 1000],
    'weight_init_stddevs': [0.02, 0.02],
    'bias_init_consts': [1., 1.],
    'dropouts': [0.25, 0.25],
    'penalty': 0.0005,
    'penalty_type': 'l2',
    'batch_size': 128,
    'nb_epoch': 50,
    'learning_rate': 0.0008,
    'fit_transformers': deepchem.trans.CoulombFitTransformer
}
hps['rf_regression'] = {'n_estimators': 500}
hps['graphconvreg'] = {
    'batch_size': 128,
    'nb_epoch': 20,
    'learning_rate': 0.0005,
    'n_filters': 128,
    'n_fully_connected_nodes': 256,
    'seed': 123
}
hps['siamese'] = {
    'n_pos': 1,
    'n_neg': 1,
    'test_batch_size': 128,
    'n_filters': [64, 128, 64],
    'n_fully_connected_nodes': [128],
    'nb_epochs': 1,
    'n_train_trials': 2000,
    'n_eval_trials': 20,
    'learning_rate': 1e-4
}
hps['res'] = {
    'n_pos': 1,
    'n_neg': 1,
    'test_batch_size': 128,
    'n_filters': [64, 128, 64],
    'n_fully_connected_nodes': [128],
    'max_depth': 3,
    'nb_epochs': 1,
    'n_train_trials': 2000,
    'n_eval_trials': 20,
    'learning_rate': 1e-4
}
hps['attn'] = {
    'n_pos': 1,
    'n_neg': 1,
    'test_batch_size': 128,
    'n_filters': [64, 128, 64],
    'n_fully_connected_nodes': [128],
    'max_depth': 3,
    'nb_epochs': 1,
    'n_train_trials': 2000,
    'n_eval_trials': 20,
    'learning_rate': 1e-4
}
