# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:31:24 2017

@author: Zhenqin Wu
"""

import torch
import numpy as np
from torch_model import TorchMultitaskModel


class TorchMultitaskRegression(TorchMultitaskModel):

  def __init__(self, n_tasks, n_features, **kwargs):
    """Constructs the computational graph.

    This function constructs the computational graph for the model. It relies
    subclassed methods (build/cost) to construct specific graphs.

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_features: int
      Number of features.
    n_classes: int
      Number of classes if this is for classification.
    """
    # Save hyperparameters
    self.n_tasks = n_tasks
    self.n_features = n_features
    super(TorchMultitaskRegression, self).__init__(**kwargs)

  def build(self):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """

    layer_sizes = self.layer_sizes
    weight_init_stddevs = self.weight_init_stddevs
    bias_init_consts = self.bias_init_consts
    dropouts = self.dropouts
    lengths_set = {
        len(layer_sizes),
        len(weight_init_stddevs),
        len(bias_init_consts),
        len(dropouts),
    }
    assert len(lengths_set) == 1, 'All layer params must have same length.'
    n_layers = lengths_set.pop()
    assert n_layers > 0, 'Must have some layers defined.'

    prev_layer_size = self.n_features
    self.W_list = []
    self.b_list = []
    for i in range(n_layers):
      W_init = np.random.normal(0, weight_init_stddevs[i],
                                (prev_layer_size, layer_sizes[i]))
      W_init = torch.cuda.FloatTensor(W_init)
      self.W_list.append(torch.autograd.Variable(W_init, requires_grad=True))
      b_init = np.full((layer_sizes[i],), bias_init_consts[i])
      b_init = torch.cuda.FloatTensor(b_init)
      self.b_list.append(torch.autograd.Variable(b_init, requires_grad=True))
      prev_layer_size = layer_sizes[i]

    self.task_W_list = []
    self.task_b_list = []
    for i in range(self.n_tasks):
      W_init = np.random.normal(0, weight_init_stddevs[-1],
                                (prev_layer_size, 1))
      W_init = torch.cuda.FloatTensor(W_init)
      self.task_W_list.append(
          torch.autograd.Variable(W_init, requires_grad=True))
      b_init = np.full((1,), bias_init_consts[-1])
      b_init = torch.cuda.FloatTensor(b_init)
      self.task_b_list.append(
          torch.autograd.Variable(b_init, requires_grad=True))
    self.trainables = self.W_list + self.b_list + self.task_W_list + self.task_b_list
    self.regularizaed_variables = self.W_list + self.task_W_list

  def forward(self, X, training=False):
    for i, W in enumerate(self.W_list):
      X = X.mm(W)
      X += self.b_list[i].unsqueeze(0).expand_as(X)
      X = torch.nn.ReLU()(X)
      if training:
        X = torch.nn.Dropout(p=self.dropouts[i])(X)
    outputs = []
    for i, W in enumerate(self.task_W_list):
      output = X.mm(W)
      output += self.task_b_list[i].unsqueeze(0).expand_as(output)
      outputs.append(output)
    return outputs

  def cost(self, logit, label, weight):
    loss = []
    loss_func = torch.nn.MSELoss()
    for i in range(logit.size()[0]):
      loss.append(loss_func(logit[i], label[i]).mul(weight[i]))
    loss = torch.cat(loss).mean()
    return loss

  def predict_on_batch(self, X_batch):
    X_batch = torch.autograd.Variable(torch.cuda.FloatTensor(X_batch))
    outputs = self.forward(X_batch, training=False)
    y_pred_batch = torch.stack(outputs, 1).data.cpu().numpy()[:]
    y_pred_batch = np.squeeze(y_pred_batch, axis=2)
    return y_pred_batch

  def predict_proba_on_batch(self, X_batch):
    raise NotImplementedError('Regression models cannot predict probability')
