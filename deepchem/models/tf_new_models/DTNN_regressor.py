#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:42:40 2017

@author: zqwu
"""
import os
import numpy as np
import tensorflow as tf
import sklearn.metrics
import tempfile
from deepchem.models.tf_new_models.multitask_regressor import MultitaskGraphRegressor


class DTNNRegressor(MultitaskGraphRegressor):

  def __init__(self,
               model,
               n_tasks=1,
               logdir=None,
               batch_size=50,
               final_loss='weighted_L2',
               learning_rate=.001,
               optimizer_type="adam",
               learning_rate_decay_time=1000,
               beta1=.9,
               beta2=.999,
               pad_batches=True,
               verbose=True):
    self.n_tasks  = n_tasks
    self.verbose = verbose
    self.n_tasks = n_tasks
    self.final_loss = final_loss
    self.model = model
    self.sess = tf.Session(graph=self.model.graph)
    if logdir is not None:
      if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
      logdir = tempfile.mkdtemp()
    self.logdir = logdir

    with self.model.graph.as_default():
      # Extract model info 
      self.batch_size = batch_size
      self.pad_batches = pad_batches
      # Get graph topology for x
      self.graph_topology = self.model.get_graph_topology()

      # Building outputs
      self.outputs = self.build()
      self.loss_op = self.add_training_loss(self.final_loss, self.outputs)

      self.learning_rate = learning_rate
      self.T = learning_rate_decay_time
      self.optimizer_type = optimizer_type

      self.optimizer_beta1 = beta1
      self.optimizer_beta2 = beta2

      # Set epsilon
      self.epsilon = 1e-7
      self.add_optimizer()

      # Initialize
      self.init_fn = tf.global_variables_initializer()
      self.sess.run(self.init_fn)

      # Path to save checkpoint files, which matches the
      # replicated supervisor's default path.
      self._save_path = os.path.join(logdir, 'model.ckpt')

  def build(self):
    # Create target inputs
    self.label_placeholder = tf.placeholder(
        dtype='float32', shape=(None, self.n_tasks), name="label_placeholder")
    self.weight_placeholder = tf.placeholder(
        dtype='float32', shape=(None, self.n_tasks), name="weight_placholder")

    outputs = self.model.return_outputs()
    return outputs
