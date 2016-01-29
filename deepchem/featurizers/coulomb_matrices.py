"""
Generate coulomb matrices for molecules.

See Montavon et al., _New Journal of Physics_ __15__ (2013) 095003.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "BSD 3-clause"

import numpy as np

from rdkit import Chem

from deepchem.featurizers import Featurizer
from deepchem.utils import pad_array


class CoulombMatrix(Featurizer):
   """
   Calculate Coulomb matrices for molecules.

   Parameters
   ----------
   max_atoms : int
       Maximum number of atoms for any molecule in the dataset. Used to
       pad the Coulomb matrix.
   remove_hydrogens : bool, optional (default True)
       Whether to remove hydrogens before constructing Coulomb matrix.
   randomize : bool, optional (default True)
       Whether to randomize Coulomb matrices to remove dependence on atom
       index order.
   n_samples : int, optional (default 1)
       Number of random Coulomb matrices to generate if randomize is True.
   seed : int, optional
       Random seed.
   """
   conformers = True
   name = 'coulomb_matrix'

   def __init__(self, max_atoms, remove_hydrogens=True, randomize=True,
                n_samples=1, seed=None):
     self.max_atoms = int(max_atoms)
     self.remove_hydrogens = remove_hydrogens
     self.randomize = randomize
     self.n_samples = n_samples
     if seed is not None:
       seed = int(seed)
     self.seed = seed

   def _featurize(self, mol):
     """
     Calculate Coulomb matrices for molecules. If extra randomized
     matrices are generated, they are treated as if they are features
     for additional conformers.

     Since Coulomb matrices are symmetric, only the (flattened) upper
     triangular portion is returned.

     Parameters
     ----------
     mol : RDKit Mol
         Molecule.
     """
     features = self.coulomb_matrix(mol)
     features = [f[np.triu_indices_from(f)] for f in features]
     features = np.asarray(features)
     return features

   def coulomb_matrix(self, mol):
     """
     Generate Coulomb matrices for each conformer of the given molecule.

     Parameters
     ----------
     mol : RDKit Mol
         Molecule.
     """
     if self.remove_hydrogens:
       mol = Chem.RemoveHs(mol)
     n_atoms = mol.GetNumAtoms()
     z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
     rval = []
     for conf in mol.GetConformers():
       d = self.get_interatomic_distances(conf)
       m = np.zeros((n_atoms, n_atoms))
       for i in xrange(mol.GetNumAtoms()):
         for j in xrange(mol.GetNumAtoms()):
           if i == j:
             m[i, j] = 0.5 * z[i] ** 2.4
           elif i < j:
             m[i, j] = (z[i] * z[j]) / d[i, j]
             m[j, i] = m[i, j]
           else:
             continue
       if self.randomize:
         for random_m in self.randomize_coulomb_matrix(m):
           random_m = pad_array(random_m, self.max_atoms)
           rval.append(random_m)
       else:
         m = pad_array(m, self.max_atoms)
         rval.append(m)
     rval = np.asarray(rval)
     return rval

   def randomize_coulomb_matrix(self, m):
     """
     Randomize a Coulomb matrix as decribed in Montavon et al., _New Journal
     of Physics_ __15__ (2013) 095003:

         1. Compute row norms for M in a vector row_norms.
         2. Sample a zero-mean unit-variance noise vector e with dimension
            equal to row_norms.
         3. Permute the rows and columns of M with the permutation that
            sorts row_norms + e.

     Parameters
     ----------
     m : ndarray
         Coulomb matrix.
     n_samples : int, optional (default 1)
         Number of random matrices to generate.
     seed : int, optional
         Random seed.
     """
     rval = []
     row_norms = np.asarray([np.linalg.norm(row) for row in m], dtype=float)
     rng = np.random.RandomState(self.seed)
     for i in xrange(self.n_samples):
       e = rng.normal(size=row_norms.size)
       p = np.argsort(row_norms + e)
       new = m[p][:, p]  # permute rows first, then columns
       rval.append(new)
     return rval

   @staticmethod
   def get_interatomic_distances(conf):
     """
     Get interatomic distances for atoms in a molecular conformer.

     Parameters
     ----------
     conf : RDKit Conformer
         Molecule conformer.
     """
     n_atoms = conf.GetNumAtoms()
     coords = [conf.GetAtomPosition(i) for i in xrange(n_atoms)]
     d = np.zeros((n_atoms, n_atoms), dtype=float)
     for i in xrange(n_atoms):
       for j in xrange(n_atoms):
         if i < j:
           d[i, j] = coords[i].Distance(coords[j])
           d[j, i] = d[i, j]
         else:
           continue
     return d
