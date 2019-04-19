#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:31:47 2017

@author: zqwu
@contributors: ytz

"""
import os
import numpy as np
import json
import scipy.optimize
import tensorflow as tf

import deepchem as dc

from deepchem.models.tensorgraph.layers import Dense, Concat, WeightedError, Stack, Layer, ANIFeat, Exp
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature, Dropout, WeightDecay, ReduceSum, Reshape
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.graph_layers import DTNNEmbedding
from deepchem.models.tensorgraph.symmetry_functions import DistanceMatrix, \
    DistanceCutoff, RadialSymmetry, AngularSymmetry, AngularSymmetryMod, \
    BPFeatureMerge, BPGather, AtomicDifferentiatedDense


class BPSymmetryFunctionRegression(TensorGraph):

  def __init__(self,
               n_tasks,
               max_atoms,
               n_feat=96,
               layer_structures=[128, 64],
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    max_atoms: int
      Maximum number of atoms in the dataset
    n_hidden: int, optional
      Number of hidden units in the readout function
    """
    self.n_tasks = n_tasks
    self.max_atoms = max_atoms
    self.n_feat = n_feat
    self.layer_structures = layer_structures

    super(BPSymmetryFunctionRegression, self).__init__(**kwargs)

    self._build_graph()

  def _build_graph(self):
    self.atom_flags = Feature(shape=(None, self.max_atoms * self.max_atoms))
    self.atom_feats = Feature(shape=(None, self.max_atoms * self.n_feat))

    reshaped_atom_feats = Reshape(
        in_layers=[self.atom_feats], shape=(-1, self.max_atoms, self.n_feat))
    reshaped_atom_flags = Reshape(
        in_layers=[self.atom_flags], shape=(-1, self.max_atoms, self.max_atoms))

    previous_layer = reshaped_atom_feats

    Hiddens = []
    for n_hidden in self.layer_structures:
      Hidden = Dense(
          out_channels=n_hidden,
          activation_fn=tf.nn.tanh,
          in_layers=[previous_layer])
      Hiddens.append(Hidden)
      previous_layer = Hiddens[-1]

    regression = Dense(
        out_channels=1 * self.n_tasks,
        activation_fn=None,
        in_layers=[Hiddens[-1]])
    output = BPGather(
        self.max_atoms, in_layers=[regression, reshaped_atom_flags])
    self.add_output(output)

    label = Label(shape=(None, self.n_tasks, 1))
    loss = ReduceSum(L2Loss(in_layers=[label, output]))
    weights = Weights(shape=(None, self.n_tasks))

    weighted_loss = WeightedError(in_layers=[loss, weights])
    self.set_loss(weighted_loss)

  def compute_features_on_batch(self, X_b):
    flags = np.sign(np.array(X_b[:, :, 0]))
    atom_flags = np.stack([flags] * self.max_atoms, axis=2) * \
                 np.stack([flags] * self.max_atoms, axis=1)
    atom_feats = np.array(X_b[:, :, 1:], dtype=np.float32)
    return [atom_feats, atom_flags]

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b

        atom_feats, atom_flags = self.compute_features_on_batch(X_b)
        atom_feats = atom_feats.reshape(-1, self.max_atoms * self.n_feat)
        atom_flags = atom_flags.reshape(-1, self.max_atoms * self.max_atoms)
        feed_dict[self.atom_feats] = atom_feats
        feed_dict[self.atom_flags] = atom_flags

        yield feed_dict

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    tensors = dict()
    for layer, column in zip(self.features, feature_columns):
      feature_col = tf.feature_column.input_layer(features, [column])
      if feature_col.dtype != column.dtype:
        feature_col = tf.cast(feature_col, column.dtype)
      tensors[layer] = feature_col

      if weight_column is not None:
        tensors[self.task_weights[0]] = tf.feature_column.input_layer(
            features, [weight_column])
      if labels is not None:
        tensors[self.labels[0]] = labels

    return tensors


class ANIRegression(TensorGraph):

  def __init__(self,
               n_tasks,
               max_atoms,
               exp_loss=False,
               activation_fn='ani',
               layer_structures=[128, 64],
               atom_number_cases=[1, 6, 7, 8, 16],
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    max_atoms: int
      Maximum number of atoms in the dataset
    n_hidden: int, optional
      Number of hidden units in the readout function
    """
    self.n_tasks = n_tasks
    self.max_atoms = max_atoms
    self.exp_loss = exp_loss
    self.activation_fn = activation_fn
    self.layer_structures = layer_structures
    self.atom_number_cases = atom_number_cases
    super(ANIRegression, self).__init__(**kwargs)

    # (ytz): this is really dirty but needed for restoring models
    self._kwargs = {
        "n_tasks": n_tasks,
        "max_atoms": max_atoms,
        "layer_structures": layer_structures,
        "atom_number_cases": atom_number_cases
    }

    self._kwargs.update(kwargs)

    self.build_graph()
    self.grad = None

  def save(self):
    self.grad = None  # recompute grad on restore
    super(ANIRegression, self).save()

  def build_grad(self):
    self.grad = tf.gradients(self.outputs, self.atom_feats)

  def compute_grad(self, dataset, upper_lim=1):
    """
    Computes a batched gradients given an input dataset.

    Parameters
    ----------
    dataset: dc.Dataset
      dataset-like object whose X values will be used to compute
      gradients from
    upper_lim: int
      subset of dataset used.

    Returns
    -------
    np.array
      Gradients of the input of shape (max_atoms, 4). Note that it is up to
      the end user to slice this matrix into the correct shape, since it's very
      likely the derivatives with respect to the atomic numbers are zero.

    """
    with self._get_tf("Graph").as_default():
      if not self.built:
        self.build()
      if not self.grad:
        self.build_grad()

      feed_dict = dict()
      X = dataset.X
      flags = np.sign(np.array(X[:upper_lim, :, 0]))
      atom_flags = np.stack([flags]*self.max_atoms, axis=2)*\
          np.stack([flags]*self.max_atoms, axis=1)
      feed_dict[self.atom_flags] = atom_flags.reshape(
          -1, self.max_atoms * self.max_atoms)
      atom_numbers = np.array(X[:upper_lim, :, 0], dtype=int)
      feed_dict[self.atom_numbers] = atom_numbers
      atom_feats = np.array(X[:upper_lim, :, :], dtype=float)
      feed_dict[self.atom_feats] = atom_feats.reshape(-1, self.max_atoms * 4)
      return self.session.run([self.grad], feed_dict=feed_dict)

  def pred_one(self, X, atomic_nums, constraints=None):
    """
    Makes an energy prediction for a set of atomic coordinates.

    Parameters
    ----------
    X: np.array
      numpy array of shape (a, 3) where a <= max_atoms and
      dtype is float-like
    atomic_nums: np.array
      numpy array of shape (a,) where a is the same as that of X.
    constraints: unused
      This parameter is mainly for compatibility purposes for scipy optimize

    Returns
    -------
    float
      Predicted energy. Note that the meaning of the returned value is
      dependent on the training y-values both in semantics (relative vs absolute)
      and units (kcal/mol vs Hartrees)

    """
    num_atoms = atomic_nums.shape[0]
    X = X.reshape((num_atoms, 3))
    A = atomic_nums.reshape((atomic_nums.shape[0], 1))
    Z = np.zeros((self.max_atoms, 4))
    Z[:X.shape[0], 1:X.shape[1] + 1] = X
    Z[:A.shape[0], :A.shape[1]] = A
    X = Z
    dd = dc.data.NumpyDataset(
        np.array(X).reshape((1, self.max_atoms, 4)), np.zeros((1, 1)),
        np.ones((1, 1)))
    return self.predict(dd)[0]

  def grad_one(self, X, atomic_nums, constraints=None):
    """
    Computes gradients for that of a single structure.

    Parameters
    ----------
    X: np.array
      numpy array of shape (a, 3) where a <= max_atoms and
      dtype is float-like
    atomic_nums: np.array
      numpy array of shape (a,) where a is the same as that of X.
    constraints: np.array
      numpy array of indices of X used for constraining a subset
      of the atoms of the molecule.

    Returns
    -------
    np.array
      derivatives of the same shape and type as input parameter X.

    """
    num_atoms = atomic_nums.shape[0]
    X = X.reshape((num_atoms, 3))
    A = atomic_nums.reshape((atomic_nums.shape[0], 1))
    Z = np.zeros((self.max_atoms, 4))
    Z[:X.shape[0], 1:X.shape[1] + 1] = X
    Z[:A.shape[0], :A.shape[1]] = A
    X = Z
    inp = np.array(X).reshape((1, self.max_atoms, 4))
    dd = dc.data.NumpyDataset(inp, np.array([1]), np.array([1]))
    res = self.compute_grad(dd)[0][0]
    res = res.reshape(self.max_atoms, 4)
    res = res[:num_atoms, 1:]

    if constraints is not None:
      for idx in constraints:
        res[idx][0] = 0
        res[idx][1] = 0
        res[idx][2] = 0

    return res.reshape((num_atoms * 3,))

  def minimize_structure(self, X, atomic_nums, constraints=None):
    """
    Minimizes a structure, as defined by a set of coordinates and their atomic
    numbers.

    Parameters
    ----------
    X: np.array
      numpy array of shape (a, 3) where a <= max_atoms and
      dtype is float-like
    atomic_nums: np.array
      numpy array of shape (a,) where a is the same as that of X.

    Returns
    -------
    np.array
      minimized coordinates of the same shape and type as input parameter X.

    """
    num_atoms = atomic_nums.shape[0]

    res = scipy.optimize.minimize(
        self.pred_one,
        X,
        args=(atomic_nums, constraints),
        jac=self.grad_one,
        method="BFGS",
        tol=1e-6,
        options={'disp': True})

    return res.x.reshape((num_atoms, 3))

  def build_graph(self):

    self.atom_numbers = Feature(shape=(None, self.max_atoms), dtype=tf.int32)
    self.atom_flags = Feature(shape=(None, self.max_atoms * self.max_atoms))
    self.atom_feats = Feature(shape=(None, self.max_atoms * 4))

    reshaped_atom_flags = Reshape(
        in_layers=[self.atom_flags], shape=(-1, self.max_atoms, self.max_atoms))
    reshaped_atom_feats = Reshape(
        in_layers=[self.atom_feats], shape=(-1, self.max_atoms, 4))

    previous_layer = ANIFeat(
        in_layers=reshaped_atom_feats, max_atoms=self.max_atoms)

    self.featurized = previous_layer

    Hiddens = []
    for n_hidden in self.layer_structures:
      Hidden = AtomicDifferentiatedDense(
          self.max_atoms,
          n_hidden,
          self.atom_number_cases,
          activation=self.activation_fn,
          in_layers=[previous_layer, self.atom_numbers])
      Hiddens.append(Hidden)
      previous_layer = Hiddens[-1]

    regression = Dense(
        out_channels=1 * self.n_tasks,
        activation_fn=None,
        in_layers=[Hiddens[-1]])
    output = BPGather(
        self.max_atoms, in_layers=[regression, reshaped_atom_flags])
    self.add_output(output)

    label = Label(shape=(None, self.n_tasks, 1))
    loss = ReduceSum(L2Loss(in_layers=[label, output]))
    weights = Weights(shape=(None, self.n_tasks))

    weighted_loss = WeightedError(in_layers=[loss, weights])
    if self.exp_loss:
      weighted_loss = Exp(in_layers=[weighted_loss])
    self.set_loss(weighted_loss)

  def compute_features_on_batch(self, X_b):
    flags = np.sign(np.array(X_b[:, :, 0]))
    atom_flags = np.stack([flags]*self.max_atoms, axis=2)*\
            np.stack([flags]*self.max_atoms, axis=1)
    atom_numbers = np.array(X_b[:, :, 0], dtype=np.int32)
    atom_feats = np.array(X_b[:, :, :], dtype=np.float32)

    return [atom_feats, atom_numbers, atom_flags]

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b

        atom_feats, atom_numbers, atom_flags = self.compute_features_on_batch(
            X_b)
        atom_feats = atom_feats.reshape(-1, self.max_atoms * 4)
        atom_flags = atom_flags.reshape(-1, self.max_atoms * self.max_atoms)
        feed_dict[self.atom_feats] = atom_feats
        feed_dict[self.atom_numbers] = atom_numbers
        feed_dict[self.atom_flags] = atom_flags
        yield feed_dict

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    tensors = dict()
    for layer, column in zip(self.features, feature_columns):
      feature_col = tf.feature_column.input_layer(features, [column])
      if feature_col.dtype != column.dtype:
        feature_col = tf.cast(feature_col, column.dtype)
      tensors[layer] = feature_col

      if weight_column is not None:
        tensors[self.task_weights[0]] = tf.feature_column.input_layer(
            features, [weight_column])
      if labels is not None:
        tensors[self.labels[0]] = labels

    return tensors
