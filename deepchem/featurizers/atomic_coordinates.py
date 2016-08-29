"""
Atomic coordinate featurizer.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL v2.1+"

import numpy as np
from deepchem.featurizers import Featurizer

class AtomicCoordinates(Featurizer):
  """
  Nx3 matrix of Cartestian coordinates [Angstrom]
  """
  name = ['atomic_coordinates']

  def _featurize(self, mol):
    """
    Calculate atomic coodinates.

    Parameters
    ----------
    mol : RDKit Mol
          Molecule.
    """

    N = mol.GetNumAtoms()
    coords = np.zeros((N,3))

    # RDKit stores atomic coordinates in Angstrom. Atomic unit of length is the
    # bohr (1 bohr = 0.529177 Angstrom). Converting units makes gradient calculation
    # consistent with most QM software packages.
    coords_in_bohr = [mol.GetConformer(0).GetAtomPosition(i).__div__(0.52917721092)
                      for i in xrange(N)]

    for atom in xrange(N):
       coords[atom,0] = coords_in_bohr[atom].x
       coords[atom,1] = coords_in_bohr[atom].y
       coords[atom,2] = coords_in_bohr[atom].z

    coords = [coords]
    return coords

