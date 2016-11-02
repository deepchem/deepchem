"""
Basic molecular features.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "LGPL v2.1+"

from rdkit.Chem import Descriptors
from deepchem.feat import Featurizer


class MolecularWeight(Featurizer):
  """
  Molecular weight.
  """
  name = ['mw', 'molecular_weight']

  def _featurize(self, mol):
    """
    Calculate molecular weight.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    wt = Descriptors.ExactMolWt(mol)
    wt = [wt]
    return wt


class RDKitDescriptors(Featurizer):
  """
  RDKit descriptors.

  See http://rdkit.org/docs/GettingStartedInPython.html
  #list-of-available-descriptors.
  """
  name = 'descriptors'

  def __init__(self):
    self.descriptors = []
    for descriptor, function in Descriptors.descList:
      self.descriptors.append(descriptor)

  def _featurize(self, mol):
    """
    Calculate RDKit descriptors.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    rval = []
    for _, function in Descriptors.descList:
      rval.append(function(mol))
    return rval
