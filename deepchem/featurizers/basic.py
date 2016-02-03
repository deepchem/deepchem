"""
Basic molecular features.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "BSD 3-clause"

from rdkit.Chem import Descriptors

from deepchem.featurizers import Featurizer


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
        self.functions = []
        for descriptor, function in Descriptors.descList:
            self.descriptors.append(descriptor)
            self.functions.append(function)

    def _featurize(self, mol):
        """
        Calculate RDKit descriptors.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        rval = []
        for function in self.functions:
            rval.append(function(mol))
        return rval
