#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:31:47 2017

@author: zqwu
"""
import numpy as np
import tensorflow as tf

from deepchem.models.tensorgraph.layers import Dense, Concat, WeightedError
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.graph_layers import DTNNEmbedding
from deepchem.models.tensorgraph.symmetry_functions import DistanceMatrix, \
    DistanceCutoff, RadialSymmetry, AngularSymmetry, AngularSymmetryMod, \
    BPFeatureMerge, BPGather, AtomicDifferentiatedDense


class BPSymmetryFunctionRegression(TensorGraph):

  def __init__(self, n_tasks, max_atoms, n_hidden=40, n_embedding=10, **kwargs):
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
    self.n_hidden = n_hidden
    self.n_embedding = n_embedding
    super(BPSymmetryFunctionRegression, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    self.atom_numbers = Feature(shape=(None, self.max_atoms), dtype=tf.int32)
    self.atom_flags = Feature(shape=(None, self.max_atoms, self.max_atoms))
    self.atom_coordinates = Feature(shape=(None, self.max_atoms, 3))

    distance_matrix = DistanceMatrix(
        self.max_atoms, in_layers=[self.atom_coordinates, self.atom_flags])
    distance_cutoff = DistanceCutoff(
        self.max_atoms,
        cutoff=6 / 0.52917721092,
        in_layers=[distance_matrix, self.atom_flags])
    radial_symmetry = RadialSymmetry(
        self.max_atoms, in_layers=[distance_cutoff, distance_matrix])
    angular_symmetry = AngularSymmetry(
        self.max_atoms,
        in_layers=[distance_cutoff, distance_matrix, self.atom_coordinates])
    atom_embedding = DTNNEmbedding(
        n_embedding=self.n_embedding, in_layers=[self.atom_numbers])

    feature_merge = BPFeatureMerge(
        self.max_atoms,
        in_layers=[
            atom_embedding, radial_symmetry, angular_symmetry, self.atom_flags
        ])

    Hidden = Dense(
        out_channels=self.n_hidden,
        activation_fn=tf.nn.tanh,
        in_layers=[feature_merge])
    Hidden2 = Dense(
        out_channels=self.n_hidden,
        activation_fn=tf.nn.tanh,
        in_layers=[Hidden])
    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      regression = Dense(
          out_channels=1, activation_fn=None, in_layers=[Hidden2])
      output = BPGather(self.max_atoms, in_layers=[regression, self.atom_flags])
      self.add_output(output)

      label = Label(shape=(None, 1))
      self.labels_fd.append(label)
      cost = L2Loss(in_layers=[label, output])
      costs.append(cost)

    all_cost = Concat(in_layers=costs, axis=0)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    for epoch in range(epochs):
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
        feed_dict[self.atom_numbers] = np.array(X_b[:, :, 0], dtype=int)
        feed_dict[self.atom_coordinates] = np.array(X_b[:, :, 1:], dtype=float)
        yield feed_dict

class ANIRegression(TensorGraph):

  def __init__(self, 
               n_tasks, 
               max_atoms, 
               n_hidden=40, 
               n_embedding=10, 
               atom_number_cases=[1, 6, 7, 8],
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
    self.n_hidden = n_hidden
    self.n_embedding = n_embedding
    self.atom_number_cases = atom_number_cases
    super(ANIRegression, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    self.atom_numbers = Feature(shape=(None, self.max_atoms), dtype=tf.int32)
    self.atom_flags = Feature(shape=(None, self.max_atoms, self.max_atoms))
    self.atom_coordinates = Feature(shape=(None, self.max_atoms, 3))

    distance_matrix = DistanceMatrix(
        self.max_atoms, in_layers=[self.atom_coordinates, self.atom_flags])
    distance_cutoff_radial = DistanceCutoff(
        self.max_atoms,
        cutoff=4.6 / 0.52917721092,
        in_layers=[distance_matrix, self.atom_flags])
    distance_cutoff_angular = DistanceCutoff(
        self.max_atoms,
        cutoff=3.1 / 0.52917721092,
        in_layers=[distance_matrix, self.atom_flags])
    radial_symmetry = RadialSymmetry(
        self.max_atoms,
        atomic_number_differentiated=True,
        atom_numbers=self.atom_number_cases,
        in_layers=[distance_cutoff_radial, distance_matrix, self.atom_numbers])
    angular_symmetry = AngularSymmetryMod(
        self.max_atoms,
        atomic_number_differentiated=True,
        atom_numbers=self.atom_number_cases,
        in_layers=[distance_cutoff_angular, distance_matrix, self.atom_coordinates, self.atom_numbers])
    atom_embedding = DTNNEmbedding(
        n_embedding=0, in_layers=[self.atom_numbers])

    feature_merge = BPFeatureMerge(
        self.max_atoms,
        in_layers=[
            atom_embedding, radial_symmetry, angular_symmetry, self.atom_flags
        ])

    Hidden = AtomicDifferentiatedDense(
        self.max_atoms,
        self.n_hidden,
        self.atom_number_cases,
        activation='tanh',
        in_layers=[feature_merge, self.atom_numbers])
    
    Hidden2 = AtomicDifferentiatedDense(
        self.max_atoms,
        self.n_hidden,
        self.atom_number_cases,
        activation='tanh',
        in_layers=[Hidden, self.atom_numbers])
    
    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      regression = Dense(
          out_channels=1, activation_fn=None, in_layers=[Hidden2])
      output = BPGather(self.max_atoms, in_layers=[regression, self.atom_flags])
      self.add_output(output)

      label = Label(shape=(None, 1))
      self.labels_fd.append(label)
      cost = L2Loss(in_layers=[label, output])
      costs.append(cost)

    all_cost = Concat(in_layers=costs, axis=0)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    for epoch in range(epochs):
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
        feed_dict[self.atom_numbers] = np.array(X_b[:, :, 0], dtype=int)
        feed_dict[self.atom_coordinates] = np.array(X_b[:, :, 1:], dtype=float)
        yield feed_dict