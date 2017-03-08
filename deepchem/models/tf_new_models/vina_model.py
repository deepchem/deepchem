"""
Implements Autodock Vina's pose-generation in tensorflow.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import tensorflow as tf
from deepchem.models import Model
from deepchem.nn import model_ops

def cutoff(d):
  """Truncates interactions that are too far away."""
  return tf.cond(d < 8, d, 0)

def gauss_1(d):
  """Computes first Gaussian interaction term.

  Note that d must be in Angstrom
  """
  return tf.exp(-(d/0.5)**2)

def gauss_2(d):
  """Computes second Gaussian interaction term.

  Note that d must be in Angstrom.
  """
  return tf.exp(-((d-3)/2)^2)


def repulsion(d):
  """Computes repulsion interaction term."""
  return tf.cond(d < 0, d**2, 0)

def hydrophobic(d):
  """Compute hydrophobic interaction term."""
  return tf.cond(d < 0.5, 1,
                 tf.cond(d < 1.5, 1.5 - d,  0))

def hbond(d):
  """Computes hydrogen bond term."""
  return tf.cond(d < -0.7, 1,
                 tf.cond(d < 0, (1.0/0.7)(0-d), 0))

def g(c, w, Nrot):
  """Nonlinear function mapping interactions to free energy."""
  return c/(1 + w*Nrot)
  

class VinaModel(Model):

  def __init__(self,
               model,
               logdir=None,
               batch_size=50):
    """Vina models.

    .. math:: c = \sum_{i < j} f_{t_i,t_j}(r_{ij})

    Over all pairs of atoms that can move relative to one-another. :math:`t_i` is the
    atomtype of atom :math:`i`.

    Can view as

    .. math:: c = c_\textrm{inter} + c_\textrm{intra}

    depending on whether atoms can move relative to one another. Free energey is
    predicted only from :math:`c_\textrm{inter}`. Let :math:`R_t` be the Van der Waal's radius of
    atom of type t. Then define surface distance

    .. math:: d_{ij} = r_{ij} - R_{t_i} - R_{t_j}

    Then the energy term is

    .. math:: f_{t_i,t_j}(r_{ij}) = \textrm{cutoff}(d_{ij}, h_{t_i,t_j}(d_{ij}))

    where
  
    .. math:: \textrm{cutoff}(d, x) = \begin{cases} x & d < 8 \textrm{ Angstrom} \\ 0 & \textrm{otherwise} \end{cases}

    The inner function can be further broken down into a sum of terms

    .. math:: h_{t_i,t_j}(d) = w_1\textrm{gauss}_1(d) + w_2\textrm{gauss}_2(d) + w_3\textrm{repulsion}(d) + w_4\textrm{hydrophobic}(d) + w_5\textrm{hbond}(d)

    these terms are defined as follows (all constants are in Angstroms):

    .. math:: 
         \textrm{gauss}_1(d) = \exp(-(d/(0.5))^2)
         \textrm{gauss}_2(d) = \exp(-((d-3)/(2))^2)
         \textrm{repulsion}(d) = \begin{cases} d^2 & d < 0 \\ 0 & d \geq 0 \end{cases}
         \textrm{hydrophobic}(d) = \begin{cases} 1 & d < 0.5 \\ 1.5 - d & \textrm{otherwise} \\ 0 & d > 1.5 \end{cases}
         \textrm{hbond}(d) = \begin{cases} 1 & d < -0.7 \\ (1.0/.7)(0 - d) & \textrm{otherwise} \\ 0 & d > 0 \end{cases}

    The free energy of binding is computed as a function of the intermolecular interactions

    ..math:: s = g(c_\textrm{inter})

    This function is defined as

    ..math:: g(c) = \frac{c}{1 + wN_\textrm{rot}}

    Where :math:`w` is a weight parameter and :math:`N_\textrm{rot}` is the number of
    rotatable bonds between heavy atoms in the ligand.

    Gradients are taken backwards through the binding-free energy function with
    respect to the position of the ligand and with respect to the torsions of
    rotatable bonds and flexible ligands.

    TODO(rbharath): It's not clear to me how the effect of the torsions on the :math:`d_{ij}` is
    computed. Is there a way to get distances from torsions?

    The idea is that mutations are applied to the ligand, and then gradient descent is
    used to optimize starting from the initial structure. The code to compute the mutations
    is specified

    https://github.com/mwojcikowski/smina/blob/master/src/lib/mutate.cpp

    Seems to do random quaternion rotations of the ligand. It's not clear to me yet
    how the flexible and rotatable bonds are handled for the system.

    Need to know an initial search space for the compound. Typically a cubic
    binding box.

    References
    ----------
    Autodock Vina Paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041641/
    Smina Paper:
    http://pubs.acs.org/doi/pdf/10.1021/ci300604z

    """
    pass

