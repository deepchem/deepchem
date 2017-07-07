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


class DistanceMatrix(Layer):
  """ TensorGraph style implementation
    The same as deepchem.nn.WeaveGather
    """

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
  """ TensorGraph style implementation
    The same as deepchem.nn.WeaveGather
    """

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

  def __init__(self, max_atoms, **kwargs):
    self.max_atoms = max_atoms
    super(RadialSymmetry, self).__init__(**kwargs)

  def build(self):
    """ Parameters for the Gaussian """
    self.Rs = tf.Variable(tf.constant(0.))
    #self.ita = tf.exp(tf.Variable(tf.constant(0.)))
    self.ita = 1.
    
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ Generate Radial Symmetry Function """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    d_cutoff = in_layers[0].out_tensor
    d = in_layers[1].out_tensor
    out_tensor = tf.exp(-self.ita * tf.square(d - self.Rs)) * d_cutoff
    self.out_tensor = tf.reduce_sum(out_tensor, axis=2)


class AngularSymmetry(Layer):
  """ Angular Symmetry Function """

  def __init__(self, max_atoms, **kwargs):
    self.max_atoms = max_atoms
    super(AngularSymmetry, self).__init__(**kwargs)

  def build(self):
    #self.lambd = tf.Variable(tf.constant(1.))
    self.lambd = 1.
    #self.ita = tf.exp(tf.Variable(tf.constant(0.)))
    self.ita = 1.
    #self.zeta = tf.Variable(tf.constant(0.8))
    self.zeta = 0.8

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
    vector_distances = tf.tile(tf.expand_dims(atom_coordinates, axis=2), (1,1,max_atoms,1)) - \
        tf.tile(tf.expand_dims(atom_coordinates, axis=1), (1,max_atoms,1,1))
    R_ij = tf.tile(tf.expand_dims(d, axis=3), (1, 1, 1, max_atoms))
    R_ik = tf.tile(tf.expand_dims(d, axis=2), (1, 1, max_atoms, 1))
    R_jk = tf.tile(tf.expand_dims(d, axis=1), (1, max_atoms, 1, 1))
    f_R_ij = tf.tile(tf.expand_dims(d_cutoff, axis=3), (1, 1, 1, max_atoms))
    f_R_ik = tf.tile(tf.expand_dims(d_cutoff, axis=2), (1, 1, max_atoms, 1))
    f_R_jk = tf.tile(tf.expand_dims(d_cutoff, axis=1), (1, max_atoms, 1, 1))

    # Define angle theta = R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance)
    theta = tf.reduce_sum(tf.tile(tf.expand_dims(vector_distances, axis=3), (1,1,1,max_atoms,1)) * \
        tf.tile(tf.expand_dims(vector_distances, axis=2), (1,1,max_atoms,1,1)), axis=4)

    theta = tf.div(theta, R_ij * R_ik + 1e-5)

    out_tensor = tf.pow(1+self.lambd*tf.cos(theta), self.zeta) * \
        tf.exp(-self.ita*(tf.square(R_ij)+tf.square(R_ik)+tf.square(R_jk))) * \
        f_R_ij * f_R_ik * f_R_jk
    self.out_tensor = tf.reduce_sum(out_tensor, axis=[2, 3]) * \
        tf.pow(tf.constant(2.), 1-self.zeta)


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
        [
            atom_embedding, tf.expand_dims(radial_symmetry, 2),
            tf.expand_dims(angular_symmetry, 2)
        ],
        axis=2)
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
