#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:43:23 2017

@author: zqwu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from deepchem.nn import activations
from deepchem.nn import initializations
from deepchem.nn import model_ops

from deepchem.models.tensorgraph.layers import Layer
from deepchem.models.tensorgraph.layers import convert_to_layers
from deepchem.metrics import to_one_hot


class DistanceMatrix(Layer):

  def __init__(self, max_atoms, **kwargs):
    """
    Parameters
    ----------
    max_atoms: int
      Maximum number of atoms in the dataset
    """
    self.max_atoms = max_atoms
    super(DistanceMatrix, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Generate distance matrix for BPSymmetryFunction with trainable cutoff """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    max_atoms = self.max_atoms
    atom_coordinates = in_layers[0].out_tensor
    atom_flags = in_layers[1].out_tensor
    tensor1 = tf.tile(
        tf.expand_dims(atom_coordinates, axis=2), (1, 1, max_atoms, 1))
    tensor2 = tf.tile(
        tf.expand_dims(atom_coordinates, axis=1), (1, max_atoms, 1, 1))
    # Calculate pairwise distance
    d = tf.sqrt(tf.reduce_sum(tf.square(tensor1 - tensor2), axis=3))
    # Masking for valid atom index
    self.out_tensor = d * tf.to_float(atom_flags)


class DistanceCutoff(Layer):

  def __init__(self, max_atoms, cutoff=6 / 0.52917721092, **kwargs):
    """
    Parameters
    ----------
    cutoff: float, optional
      cutoff threshold for distance, in Bohr(0.53Angstrom)
    """
    self.max_atoms = max_atoms
    self.cutoff = cutoff
    super(DistanceCutoff, self).__init__(**kwargs)

  def build(self):
    self.Rc = tf.Variable(tf.constant(self.cutoff))

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Generate distance matrix for BPSymmetryFunction with trainable cutoff """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    d = in_layers[0].out_tensor
    d_flag = in_layers[1].out_tensor
    # Cutoff with threshold Rc
    d_flag = d_flag * tf.nn.relu(tf.sign(self.Rc - d))
    d = 0.5 * (tf.cos(np.pi * d / self.Rc) + 1)
    out_tensor = d * d_flag
    out_tensor = out_tensor * tf.expand_dims((1 - tf.eye(self.max_atoms)), 0)
    self.out_tensor = out_tensor


class RadialSymmetry(Layer):
  """ Radial Symmetry Function """

  def __init__(self,
               max_atoms,
               Rs_init=None,
               ita_init=None,
               atomic_number_differentiated=False,
               atom_numbers=[1, 6, 7, 8],
               **kwargs):
    self.max_atoms = max_atoms
    self.atomic_number_differentiated = atomic_number_differentiated
    self.atom_number_cases = atom_numbers
    if Rs_init is None:
      self.Rs_init = np.array([0.5, 1.17, 1.83, 2.5, 3.17, 3.83, 4.5])
      self.Rs_init = self.Rs_init / 0.52917721092
    else:
      self.Rs_init = np.array(Rs_init)
    if ita_init is None:
      self.ita_init = np.array([1.12])
    else:
      self.ita_init = np.array(ita_init)

    super(RadialSymmetry, self).__init__(**kwargs)

  def build(self):
    """ Parameters for the Gaussian """
    len_Rs = len(self.Rs_init)
    len_ita = len(self.ita_init)
    self.length = len_Rs * len_ita
    Rs_init, ita_init = np.meshgrid(self.Rs_init, self.ita_init)
    self.Rs = tf.constant(Rs_init.flatten(), dtype=tf.float32)
    self.ita = tf.constant(ita_init.flatten(), dtype=tf.float32)
    self.atom_number_embedding = tf.eye(max(self.atom_number_cases) + 1)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Generate Radial Symmetry Function """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    d_cutoff = in_layers[0].out_tensor
    d = in_layers[1].out_tensor
    if self.atomic_number_differentiated:
      atom_numbers = in_layers[2].out_tensor
      atom_number_embedded = tf.nn.embedding_lookup(self.atom_number_embedding,
                                                    atom_numbers)
    d_cutoff = tf.stack([d_cutoff] * self.length, axis=3)
    d = tf.stack([d] * self.length, axis=3)
    Rs = tf.reshape(self.Rs, (1, 1, 1, -1))
    ita = tf.reshape(self.ita, (1, 1, 1, -1))
    out_tensor = tf.exp(-ita * tf.square(d - Rs)) * d_cutoff
    if self.atomic_number_differentiated:
      out_tensors = []
      for atom_type in self.atom_number_cases:
        selected_atoms = tf.expand_dims(
            tf.expand_dims(atom_number_embedded[:, :, atom_type], axis=1),
            axis=3)
        out_tensors.append(tf.reduce_sum(out_tensor * selected_atoms, axis=2))
      self.out_tensor = tf.concat(out_tensors, axis=2)
    else:
      self.out_tensor = tf.reduce_sum(out_tensor, axis=2)


class AngularSymmetry(Layer):
  """ Angular Symmetry Function """

  def __init__(self,
               max_atoms,
               lambd_init=None,
               ita_init=None,
               zeta_init=None,
               **kwargs):
    self.max_atoms = max_atoms
    if lambd_init is None:
      self.lambd_init = np.array([1., -1.])
    else:
      self.lambd_init = np.array(lambd_init)

    if ita_init is None:
      self.ita_init = np.array([4.])
    else:
      self.ita_init = np.array(ita_init)

    if zeta_init is None:
      self.zeta_init = np.array([2., 4., 8.])
    else:
      self.zeta_init = np.array(zeta_init)

    super(AngularSymmetry, self).__init__(**kwargs)

  def build(self):
    len_lambd = len(self.lambd_init)
    len_ita = len(self.ita_init)
    len_zeta = len(self.zeta_init)
    self.length = len_lambd * len_ita * len_zeta

    lambd_init, ita_init, zeta_init = np.meshgrid(self.lambd_init,
                                                  self.ita_init, self.zeta_init)
    self.lambd = tf.constant(lambd_init.flatten(), dtype=tf.float32)
    self.ita = tf.constant(ita_init.flatten(), dtype=tf.float32)
    self.zeta = tf.constant(zeta_init.flatten(), dtype=tf.float32)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Generate Angular Symmetry Function """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    max_atoms = self.max_atoms
    d_cutoff = in_layers[0].out_tensor
    d = in_layers[1].out_tensor
    atom_coordinates = in_layers[2].out_tensor
    vector_distances = tf.tile(tf.expand_dims(atom_coordinates, axis=2), (1, 1, max_atoms, 1)) - \
                       tf.tile(tf.expand_dims(atom_coordinates, axis=1), (1, max_atoms, 1, 1))
    R_ij = tf.tile(tf.expand_dims(d, axis=3), (1, 1, 1, max_atoms))
    R_ik = tf.tile(tf.expand_dims(d, axis=2), (1, 1, max_atoms, 1))
    R_jk = tf.tile(tf.expand_dims(d, axis=1), (1, max_atoms, 1, 1))
    f_R_ij = tf.tile(tf.expand_dims(d_cutoff, axis=3), (1, 1, 1, max_atoms))
    f_R_ik = tf.tile(tf.expand_dims(d_cutoff, axis=2), (1, 1, max_atoms, 1))
    f_R_jk = tf.tile(tf.expand_dims(d_cutoff, axis=1), (1, max_atoms, 1, 1))

    # Define angle theta = R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance)
    theta = tf.reduce_sum(tf.tile(tf.expand_dims(vector_distances, axis=3), (1, 1, 1, max_atoms, 1)) * \
                          tf.tile(tf.expand_dims(vector_distances, axis=2), (1, 1, max_atoms, 1, 1)), axis=4)

    theta = tf.div(theta, R_ij * R_ik + 1e-5)

    R_ij = tf.stack([R_ij] * self.length, axis=4)
    R_ik = tf.stack([R_ik] * self.length, axis=4)
    R_jk = tf.stack([R_jk] * self.length, axis=4)
    f_R_ij = tf.stack([f_R_ij] * self.length, axis=4)
    f_R_ik = tf.stack([f_R_ik] * self.length, axis=4)
    f_R_jk = tf.stack([f_R_jk] * self.length, axis=4)

    theta = tf.stack([theta] * self.length, axis=4)
    lambd = tf.reshape(self.lambd, (1, 1, 1, 1, -1))
    zeta = tf.reshape(self.zeta, (1, 1, 1, 1, -1))
    ita = tf.reshape(self.ita, (1, 1, 1, 1, -1))

    out_tensor = tf.pow(1 + lambd * tf.cos(theta), zeta) * \
                 tf.exp(-ita * (tf.square(R_ij) + tf.square(R_ik) + tf.square(R_jk))) * \
                 f_R_ij * f_R_ik * f_R_jk
    self.out_tensor = tf.reduce_sum(out_tensor, axis=[2, 3]) * \
                      tf.pow(tf.constant(2.), 1 - tf.reshape(self.zeta, (1, 1, -1)))


class AngularSymmetryMod(Layer):
  """ Angular Symmetry Function """

  def __init__(self,
               max_atoms,
               lambd_init=None,
               ita_init=None,
               zeta_init=None,
               Rs_init=None,
               thetas_init=None,
               atomic_number_differentiated=False,
               atom_numbers=[1, 6, 7, 8],
               **kwargs):
    self.max_atoms = max_atoms
    self.atomic_number_differentiated = atomic_number_differentiated
    self.atom_number_cases = atom_numbers
    if lambd_init is None:
      self.lambd_init = np.array([1., -1.])
    else:
      self.lambd_init = np.array(lambd_init)

    if ita_init is None:
      self.ita_init = np.array([1.12])
    else:
      self.ita_init = np.array(ita_init)

    if zeta_init is None:
      self.zeta_init = np.array([4.])
    else:
      self.zeta_init = np.array(zeta_init)

    if Rs_init is None:
      self.Rs_init = np.array([0.5, 1.17, 1.83, 2.5, 3.17])
      self.Rs_init = self.Rs_init / 0.52917721092
    else:
      self.Rs_init = np.array(Rs_init)

    if thetas_init is None:
      self.thetas_init = np.array([0., 1.57, 3.14, 4.71])
    else:
      self.thetas_init = np.array(thetas_init)
    super(AngularSymmetryMod, self).__init__(**kwargs)

  def build(self):
    len_lambd = len(self.lambd_init)
    len_ita = len(self.ita_init)
    len_zeta = len(self.zeta_init)
    len_Rs = len(self.Rs_init)
    len_thetas = len(self.thetas_init)
    self.length = len_lambd * len_ita * len_zeta * len_Rs * len_thetas

    lambd_init, ita_init, zeta_init, Rs_init, thetas_init = \
      np.meshgrid(self.lambd_init, self.ita_init, self.zeta_init, self.Rs_init, self.thetas_init)
    self.lambd = tf.constant(lambd_init.flatten(), dtype=tf.float32)
    self.ita = tf.constant(ita_init.flatten(), dtype=tf.float32)
    self.zeta = tf.constant(zeta_init.flatten(), dtype=tf.float32)
    self.Rs = tf.constant(Rs_init.flatten(), dtype=tf.float32)
    self.thetas = tf.constant(thetas_init.flatten(), dtype=tf.float32)
    self.atom_number_embedding = tf.eye(max(self.atom_number_cases) + 1)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Generate Angular Symmetry Function """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    max_atoms = self.max_atoms
    d_cutoff = in_layers[0].out_tensor
    d = in_layers[1].out_tensor
    atom_coordinates = in_layers[2].out_tensor
    if self.atomic_number_differentiated:
      atom_numbers = in_layers[3].out_tensor
      atom_number_embedded = tf.nn.embedding_lookup(self.atom_number_embedding,
                                                    atom_numbers)

    vector_distances = tf.tile(tf.expand_dims(atom_coordinates, axis=2), (1, 1, max_atoms, 1)) - \
                       tf.tile(tf.expand_dims(atom_coordinates, axis=1), (1, max_atoms, 1, 1))
    R_ij = tf.tile(tf.expand_dims(d, axis=3), (1, 1, 1, max_atoms))
    R_ik = tf.tile(tf.expand_dims(d, axis=2), (1, 1, max_atoms, 1))
    f_R_ij = tf.tile(tf.expand_dims(d_cutoff, axis=3), (1, 1, 1, max_atoms))
    f_R_ik = tf.tile(tf.expand_dims(d_cutoff, axis=2), (1, 1, max_atoms, 1))

    # Define angle theta = R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance)
    theta = tf.reduce_sum(tf.tile(tf.expand_dims(vector_distances, axis=3), (1, 1, 1, max_atoms, 1)) * \
                          tf.tile(tf.expand_dims(vector_distances, axis=2), (1, 1, max_atoms, 1, 1)), axis=4)

    theta = tf.div(theta, R_ij * R_ik + 1e-5)

    R_ij = tf.stack([R_ij] * self.length, axis=4)
    R_ik = tf.stack([R_ik] * self.length, axis=4)
    f_R_ij = tf.stack([f_R_ij] * self.length, axis=4)
    f_R_ik = tf.stack([f_R_ik] * self.length, axis=4)

    theta = tf.stack([theta] * self.length, axis=4)
    lambd = tf.reshape(self.lambd, (1, 1, 1, 1, -1))
    zeta = tf.reshape(self.zeta, (1, 1, 1, 1, -1))
    ita = tf.reshape(self.ita, (1, 1, 1, 1, -1))
    Rs = tf.reshape(self.Rs, (1, 1, 1, 1, -1))
    thetas = tf.reshape(self.thetas, (1, 1, 1, 1, -1))

    out_tensor = tf.pow(1 + lambd * tf.cos(theta - thetas), zeta) * \
                 tf.exp(-ita * tf.square((R_ij + R_ik) / 2 - Rs)) * \
                 f_R_ij * f_R_ik * tf.pow(tf.constant(2.), 1 - zeta)
    if self.atomic_number_differentiated:
      out_tensors = []
      for atom_type_j in self.atom_number_cases:
        for atom_type_k in self.atom_number_cases:
          selected_atoms = tf.stack([atom_number_embedded[:, :, atom_type_j]] * max_atoms, axis=2) * \
                           tf.stack([atom_number_embedded[:, :, atom_type_k]] * max_atoms, axis=1)
          selected_atoms = tf.expand_dims(
              tf.expand_dims(selected_atoms, axis=1), axis=4)
          out_tensors.append(
              tf.reduce_sum(out_tensor * selected_atoms, axis=[2, 3]))
      self.out_tensor = tf.concat(out_tensors, axis=2)
    else:
      self.out_tensor = tf.reduce_sum(out_tensor, axis=[2, 3])


class BPFeatureMerge(Layer):

  def __init__(self, max_atoms, **kwargs):
    self.max_atoms = max_atoms
    super(BPFeatureMerge, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Merge features together """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    atom_embedding = in_layers[0].out_tensor
    radial_symmetry = in_layers[1].out_tensor
    angular_symmetry = in_layers[2].out_tensor
    atom_flags = in_layers[3].out_tensor

    out_tensor = tf.concat(
        [atom_embedding, radial_symmetry, angular_symmetry], axis=2)
    self.out_tensor = out_tensor * atom_flags[:, :, 0:1]


class BPGather(Layer):

  def __init__(self, max_atoms, **kwargs):
    self.max_atoms = max_atoms
    super(BPGather, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Merge features together """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    out_tensor = in_layers[0].out_tensor
    flags = in_layers[1].out_tensor

    out_tensor = tf.reduce_sum(out_tensor * flags[:, :, 0:1], axis=1)
    self.out_tensor = out_tensor


class AtomicDifferentiatedDense(Layer):
  """ Separate Dense module for different atoms """

  def __init__(self,
               max_atoms,
               out_channels,
               atom_number_cases=[1, 6, 7, 8],
               init='glorot_uniform',
               activation='relu',
               **kwargs):
    self.init = init  # Set weight initialization
    self.activation = activation  # Get activations
    self.max_atoms = max_atoms
    self.out_channels = out_channels
    self.atom_number_cases = atom_number_cases

    super(AtomicDifferentiatedDense, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Generate Radial Symmetry Function """
    init_fn = initializations.get(self.init)  # Set weight initialization
    activation_fn = activations.get(self.activation)
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    inputs = in_layers[0].out_tensor
    atom_numbers = in_layers[1].out_tensor
    in_channels = inputs.get_shape().as_list()[-1]
    self.W = init_fn(
        [len(self.atom_number_cases), in_channels, self.out_channels])

    self.b = model_ops.zeros((len(self.atom_number_cases), self.out_channels))
    outputs = []
    for i, atom_case in enumerate(self.atom_number_cases):
      # optimization to allow for tensorcontraction/broadcasted mmul
      # using a reshape trick. Note that the np and tf matmul behavior
      # differs when dealing with broadcasts

      a = inputs  # (i,j,k)
      b = self.W[i, :, :]  # (k, l)

      ai = tf.shape(a)[0]
      aj = tf.shape(a)[1]
      ak = tf.shape(a)[2]
      bl = tf.shape(b)[1]

      output = activation_fn(
          tf.reshape(tf.matmul(tf.reshape(a, [ai * aj, ak]), b), [ai, aj, bl]) +
          self.b[i, :])

      mask = 1 - tf.to_float(tf.cast(atom_numbers - atom_case, tf.bool))
      output = tf.reshape(output * tf.expand_dims(mask, 2), (-1, self.max_atoms,
                                                             self.out_channels))
      outputs.append(output)
    self.out_tensor = tf.add_n(outputs)

  def none_tensors(self):
    w, b, out_tensor = self.W, self.b, self.out_tensor
    self.W, self.b, self.out_tensor = None, None, None
    return w, b, out_tensor

  def set_tensors(self, tensor):
    self.W, self.b, self.out_tensor = tensor
