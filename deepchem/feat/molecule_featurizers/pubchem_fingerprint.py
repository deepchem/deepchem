import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


class PubChemFingerprint(MolecularFeaturizer):
    """PubChem Fingerprint.

    The PubChem fingerprint is a 881 bit structural key,
    which is used by PubChem for similarity searching.
    Please confirm the details in [1]_.

    References
    ----------
    .. [1] ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.pdf

    Note
    -----
    This class requires RDKit and PubChemPy to be installed.
    PubChemPy use REST API to get the fingerprint, so you need the internet access.

    Examples
    --------
    >>> import deepchem as dc
    >>> smiles = ['CCC']
    >>> featurizer = dc.feat.PubChemFingerprint()
    >>> features = featurizer.featurize(smiles)
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (881,)

    """

    def __init__(self):
        """Initialize this featurizer."""
        try:
            from rdkit import Chem  # noqa
            import pubchempy as pcp  # noqa
        except ModuleNotFoundError:
            raise ImportError("This class requires PubChemPy to be installed.")

        self.get_pubchem_compounds = pcp.get_compounds

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate PubChem fingerprint.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit Mol object

        Returns
        -------
        np.ndarray
            1D array of RDKit descriptors for `mol`. The length is 881.

        """
        try:
            from rdkit import Chem
            import pubchempy as pcp
        except ModuleNotFoundError:
            raise ImportError("This class requires PubChemPy to be installed.")
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        smiles = Chem.MolToSmiles(datapoint)
        pubchem_compound = pcp.get_compounds(smiles, 'smiles')[0]
        feature = [int(bit) for bit in pubchem_compound.cactvs_fingerprint]
        return np.asarray(feature)
