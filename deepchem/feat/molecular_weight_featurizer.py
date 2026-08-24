"""Molecular weight featurizer for DeepChem.

This featurizer computes the exact molecular weight of molecules
given as SMILES strings using RDKit.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from deepchem.feat import MolecularFeaturizer


class MolecularWeightFeaturizer(MolecularFeaturizer):
    """Compute molecular weight from SMILES.

    Examples
    --------
    >>> from deepchem.feat import MolecularWeightFeaturizer
    >>> featurizer = MolecularWeightFeaturizer()
    >>> features = featurizer.featurize(["CCO"])
    >>> features.shape
    (1, 1)
    >>> round(float(features[0][0]), 2)
    46.04
    """

    def __init__(self):
        super().__init__()

    def featurize(self, smiles_list):
        """Convert SMILES to molecular weight features.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings.

        Returns
        -------
        np.ndarray
            Shape (N, 1) array of molecular weights.
        """

        results = []

        for smi in smiles_list:

            try:
                mol = Chem.MolFromSmiles(smi)

                if mol is None:
                    results.append([0.0])
                    continue

                mw = Descriptors.ExactMolWt(mol)
                results.append([mw])

            except Exception:
                results.append([0.0])

        return np.array(results, dtype=np.float32)
