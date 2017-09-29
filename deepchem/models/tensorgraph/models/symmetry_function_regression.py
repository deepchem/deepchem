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

from deepchem.models.tensorgraph.layers import Dense, Concat, WeightedError, Stack, Layer, ANIFeat
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
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

    self.build_graph()

  def build_graph(self):
    self.atom_flags = Feature(shape=(None, self.max_atoms, self.max_atoms))
    self.atom_feats = Feature(shape=(None, self.max_atoms, self.n_feat))
    previous_layer = self.atom_feats

    Hiddens = []
    for n_hidden in self.layer_structures:
      Hidden = Dense(
          out_channels=n_hidden,
          activation_fn=tf.nn.tanh,
          in_layers=[previous_layer])
      Hiddens.append(Hidden)
      previous_layer = Hiddens[-1]

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      regression = Dense(
          out_channels=1, activation_fn=None, in_layers=[Hiddens[-1]])
      output = BPGather(self.max_atoms, in_layers=[regression, self.atom_flags])
      self.add_output(output)

      label = Label(shape=(None, 1))
      self.labels_fd.append(label)
      cost = L2Loss(in_layers=[label, output])
      costs.append(cost)

    all_cost = Stack(in_layers=costs, axis=1)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None and not predict:
          feed_dict[self.weights] = w_b

        flags = np.sign(np.array(X_b[:, :, 0]))
        feed_dict[self.atom_flags] = np.stack([flags]*self.max_atoms, axis=2)*\
            np.stack([flags]*self.max_atoms, axis=1)
        feed_dict[self.atom_feats] = np.array(X_b[:, :, 1:], dtype=float)
        yield feed_dict


class ANIRegression(TensorGraph):

  def __init__(self,
               n_tasks,
               max_atoms,
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
      feed_dict[self.atom_flags] = np.stack([flags]*self.max_atoms, axis=2)*\
          np.stack([flags]*self.max_atoms, axis=1)
      feed_dict[self.atom_numbers] = np.array(X[:upper_lim, :, 0], dtype=int)
      feed_dict[self.atom_feats] = np.array(X[:upper_lim, :, :], dtype=float)
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
        np.array(X).reshape((1, self.max_atoms, 4)), np.array(0), np.array(1))
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
    res = self.compute_grad(dd)[0][0][0]
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
    self.atom_flags = Feature(shape=(None, self.max_atoms, self.max_atoms))
    self.atom_feats = Feature(shape=(None, self.max_atoms, 4))

    previous_layer = ANIFeat(
        in_layers=self.atom_feats, max_atoms=self.max_atoms)

    self.featurized = previous_layer

    Hiddens = []
    for n_hidden in self.layer_structures:
      Hidden = AtomicDifferentiatedDense(
          self.max_atoms,
          n_hidden,
          self.atom_number_cases,
          activation='tanh',
          in_layers=[previous_layer, self.atom_numbers])
      Hiddens.append(Hidden)
      previous_layer = Hiddens[-1]

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      regression = Dense(
          out_channels=1, activation_fn=None, in_layers=[Hiddens[-1]])
      output = BPGather(self.max_atoms, in_layers=[regression, self.atom_flags])
      self.add_output(output)

      label = Label(shape=(None, 1))
      self.labels_fd.append(label)
      cost = L2Loss(in_layers=[label, output])
      costs.append(cost)

    all_cost = Stack(in_layers=costs, axis=1)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

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
          for index, label in enumerate(self.labels_fd):
            feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None and not predict:
          feed_dict[self.weights] = w_b

        flags = np.sign(np.array(X_b[:, :, 0]))
        feed_dict[self.atom_flags] = np.stack([flags]*self.max_atoms, axis=2)*\
            np.stack([flags]*self.max_atoms, axis=1)
        feed_dict[self.atom_numbers] = np.array(X_b[:, :, 0], dtype=int)
        feed_dict[self.atom_feats] = np.array(X_b[:, :, :], dtype=float)
        yield feed_dict

  def save_numpy(self):
    """
    Save to a portable numpy file. Note that this relies on the names to be consistent
    across different versions. The file is saved as save_pickle.npz under the model_dir.

    """
    path = os.path.join(self.model_dir, "save_pickle.npz")

    with self._get_tf("Graph").as_default():
      all_vars = tf.trainable_variables()
      all_vals = self.session.run(all_vars)
      save_dict = {}
      for idx, _ in enumerate(all_vars):
        save_dict[all_vars[idx].name] = all_vals[idx]

      save_dict["_kwargs"] = np.array(
          [json.dumps(self._kwargs)], dtype=np.string_)

      np.savez(path, **save_dict)

  @classmethod
  def load_numpy(cls, model_dir):
    """
    Load from a portable numpy file.

    Parameters
    ----------
    model_dir: str
      Location of the model directory.

    """
    path = os.path.join(model_dir, "save_pickle.npz")
    npo = np.load(path)

    json_blob = npo["_kwargs"][0].decode('UTF-8')

    kwargs = json.loads(json_blob)

    obj = cls(**kwargs)
    obj.build()

    all_ops = []

    g = obj._get_tf("Graph")

    with g.as_default():
      all_vars = tf.trainable_variables()
      for k in npo.keys():

        if k == "_kwargs":
          continue

        val = npo[k]
        tensor = g.get_tensor_by_name(k)
        all_ops.append(tf.assign(tensor, val))

      obj.session.run(all_ops)

    return obj
