#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:31:47 2017

@author: zqwu
@contributors: ytz

"""
import math
import threading
import os
import numpy as np
import json
import scipy.optimize
import scipy.sparse
import tensorflow as tf
import time
import multiprocessing.pool
import deepchem as dc

from deepchem.models.tensorgraph.layers import Dense, Concat, RootMeanError, Stack, Layer, ANIFeat, ShiftedExponential
from deepchem.models.tensorgraph.layers import L1Loss, L2Loss, Conditional, Reshape, Label, Weights, Feature, BPGather2
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
    loss = RootWeightedMeanError(in_layers=[all_cost, self.weights])
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
               layer_structures=[128, 64, 1],
               atom_number_cases=[1, 6, 7, 8],
               shift_exp=False,
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
    self.shift_exp = shift_exp
    super(ANIRegression, self).__init__(**kwargs)

    # (ytz): this is really dirty but needed for restoring models
    self._kwargs = {
        "n_tasks": n_tasks,
        "max_atoms": max_atoms,
        "layer_structures": layer_structures,
        "atom_number_cases": atom_number_cases,
    }

    self._kwargs.update(kwargs)

    self.build_graph()
    self.grad = None

  def save(self):
    self.grad = None  # recompute grad on restore
    super(ANIRegression, self).save()

  def build_grad(self):
    self.grad = tf.gradients(self.outputs, self.dequeue_object)

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
      feed_dict = {
          self.mode: True,
          self.dequeue_object: np.array(
              dataset.X[:upper_lim, :, :], dtype=float)
      }

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

    self.mode = Feature(dtype=tf.bool)  # if true then we need to featurize
    self.dequeue_object = Feature()  # either atom_feats or featurized

    def true_fn():
      r_feat = Reshape(
          shape=[None, self.max_atoms, 4], in_layers=[self.dequeue_object])
      r_feat.create_tensor()

      ani_feat = ANIFeat(in_layers=[r_feat], max_atoms=self.max_atoms)

      ani_feat.create_tensor()
      return ani_feat

    def false_fn():

      # dequeue yields a sparse tensor for the entire shard

      r_feat = Reshape(
        # (ytz): We don't actually want to hardcode this but this information
        # is a little hard to extract a priori
          shape=[None, self.max_atoms, 385], in_layers=[self.dequeue_object])
      r_feat.create_tensor()
      return r_feat

    self.featurized = Conditional(
        in_layers=[self.mode, self.dequeue_object],
        true_fn=true_fn,
        false_fn=false_fn)

    previous_layer = self.featurized

    Hiddens = []
    for idx, n_hidden in enumerate(self.layer_structures):

      if idx != len(self.layer_structures)-1:
        act_fn = 'gaussian'
      else:
        act_fn = 'linear'

      Hidden = AtomicDifferentiatedDense(
          self.max_atoms,
          n_hidden,
          self.atom_number_cases,
          activation=act_fn,
          in_layers=[previous_layer, self.featurized])
      Hiddens.append(Hidden)
      previous_layer = Hiddens[-1]

    self.DEBUG_LAYER = previous_layer

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      output = BPGather2(in_layers=[Hiddens[-1], self.featurized])
      self.add_output(output)
      label = Label(shape=(None, 1))
      self.labels_fd.append(label)
      cost = L2Loss(in_layers=[label, output])
      costs.append(cost)

    all_cost = Stack(in_layers=costs, axis=1)
    # self.weights = Weights(shape=(None, self.n_tasks))
    if self.shift_exp:
      print("WARNING: Using shifted exponential loss")
      loss = RootMeanError(in_layers=[all_cost])
      loss = ShiftedExponential(0.5, in_layers=loss)
    else:
      loss = RootMeanError(in_layers=[all_cost])

    self.set_loss(loss)

    # max_normalization
    # ws = tf.trainable_variables()
    # ops = []
    # for h in Hiddens:
      # ops.append(tf.assign(h.W, tf.clip_by_norm(w, 3.0, axes=1)))
    # self.maxnorm_op = ops
    self.hiddens = Hiddens

  def weight_matrices(self):
    weights = []
    for h in self.hiddens:
      weights.append(h.W)
    return weights

  def get_maxnorm_ops(self):
    try:
      return self._ops
    except AttributeError:
      with self._get_tf("Graph").as_default():
        ws = self.weight_matrices()
        ops = []
        for w in ws:
          ops.append(tf.assign(w, tf.clip_by_norm(w, 3.0, axes=1)))
        self._ops = ops
      return self._ops

  # def featurize(self, dataset, deterministic, pad_batches):
  def featurize(self, dataset, feat_dir):
    """
    Use the model to featurize the dataset.

    Parameters
    ----------
    dataset: dc.data.DiskDataset
      Dataset that we're featurizing

    feat_dir: str
      Path to store featurization data

    Returns
    -------
    dc.data.DiskDataset
      A DiskDatasets object representing the featurized data.

    """

    fp = os.path.join(dataset.data_dir, "feat")
    if os.path.exists(fp):
      dataset.feat_dataset = dc.data.DiskDataset(data_dir=fp)

    start_time = time.time()

    # (ytz): Featurization uses significantly more memory than training
    # so the batch_size should be smaller
    batch_size = self.batch_size
    shard_size = batch_size * 512

    def shard_generator(cself):

      print("Dataset needs to be featurized...")
      # (ytz): enqueue in a separate thread
      # yay for shitty thread-safe GILs on lists
      all_ybs = []

      def feed_queue():

        for bidx, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(
            batch_size=batch_size,
            deterministic=True,
            pad_batches=False)):

          # print("ENQUEUE LENGTH", bidx, len(X_b), len(y_b))

          # debugging race condition
          mode = cself.get_pre_q_input(cself.mode) # FEAT_0_PREQ
          obj = cself.get_pre_q_input(cself.dequeue_object) # FEAT_1_PREQ
          lab = cself.get_pre_q_input(cself.labels[0]) # FEAT_2_PREQ
          # weights = cself.get_pre_q_input(cself.task_weights[0]) # FEAT_3_PREQ

          cself.session.run(
              cself.input_queue, # enqueue_op
              feed_dict={
                  mode: True,
                  obj: X_b,
                  lab: np.zeros(
                      (batch_size, 1)
                  ),  # TODO(ytz): would be nice if we didn't have to feed dummys
              })

          all_ybs.append(y_b)

      t1 = threading.Thread(target=feed_queue)
      t1.start()

      batch_idx = 0

      # number of batches in this shard
      num_batches = math.ceil(len(dataset) / batch_size)

      run_time = 0
      total_time = time.time()

      # use a seperate thread to do the compute so we can overlap with the file write
      pool = multiprocessing.pool.ThreadPool(1)
      next_batch = pool.apply_async(cself.session.run, (cself.featurized,))

      print("Total number of batches:", num_batches, "Batch Size:", batch_size)

      # preallocate a shard and fill on the fly
      shard_X_pre_alloc = np.zeros((shard_size, self.max_atoms, 385), dtype=np.float32)
      shard_y_pre_alloc = np.zeros((shard_size, 1), dtype=np.float32)

      start_idx = 0

      while batch_idx < num_batches:

        X_feat = next_batch.get()  # shape (batch_size, max_atoms, feat_size)
        end_idx = start_idx + len(X_feat)

        shard_X_pre_alloc[start_idx:end_idx, :, :] = X_feat

        if batch_idx < num_batches - 1:
          next_batch = pool.apply_async(cself.session.run, (cself.featurized,))

        y_b = all_ybs[batch_idx]
        shard_y_pre_alloc[start_idx:end_idx, :] = y_b

        if end_idx == shard_size or batch_idx == num_batches - 1:

          yield shard_X_pre_alloc[:end_idx, :, :], \
                shard_y_pre_alloc[:end_idx, :], \
                None, \
                None

          start_idx = 0
        else:
          start_idx += len(X_feat)

        batch_idx += 1


      pool.close()
      t1.join()

    res = dc.data.DiskDataset.create_dataset(
        shard_generator=shard_generator(self),
        data_dir=feat_dir,
        X_is_sparse=True)

    # print(res.get_shape(), dataset.get_shape())
    # assert len(res) == len(dataset)
    return res
    # print("Finished featurization, total_time: " + strtime.time()-start_time + " seconds.")

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=False):

    assert pad_batches == False

    if predict and not deterministic:
      raise Exception(
            "Called predict with non-deterministic generator, terrible idea.")

    # print("DEFAULT_GENERATOR BATCH_SIZE", self.batch_size)

    batch_size = self.batch_size

    # way faster on un-featurized data
    dataset_size = len(dataset)

    needs_feat = True
    try:
      dataset = dataset.feat_dataset
      needs_feat = False
      print("Skipping featurization...")
    except AttributeError:
      print("Requires manual featurization")
      pass


    # print("??")
    # print("!!", len(dataset), needs_feat)
    print("--")

    for epoch in range(epochs):
      if predict:
        generator = dataset.iterbatches(
          deterministic=deterministic,
          batch_size=batch_size,
          dataset_size=dataset_size,
          load_ys=False,
          load_ws=False,
          load_ids=False)
      else:
        generator = dataset.iterbatches(
          deterministic=deterministic,
          batch_size=batch_size,
          dataset_size=dataset_size,
          load_ys=True,
          load_ws=False,
          load_ids=False)

      for (X_b, y_b, w_b, ids_b) in generator:

        # print(YIELDING)

        # print("X_b shape", X_b.shape)
        feed_dict = {}
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None and not predict:
          feed_dict[self.weights] = np.ones((self.batch_size, 1))

        # print("NEEDS_FEAT:", needs_feat)

        feed_dict[self.mode] = needs_feat
        feed_dict[self.dequeue_object] = X_b

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
  def load_numpy(cls, model_dir, override_kwargs=None):
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

    if override_kwargs:
      for k, v in override_kwargs.items():
        kwargs[k] = v

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
