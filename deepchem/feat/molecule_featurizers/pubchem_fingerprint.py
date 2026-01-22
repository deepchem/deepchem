import numpy as np
import base64
import json
import urllib.request
import urllib.parse

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


def _get_pubchem_fingerprint(smiles: str) -> np.ndarray:
    """Fetch and decode PubChem fingerprint via REST API."""

    # Build the request (POST to handle special chars in SMILES)
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/Fingerprint2D/JSON"
    data = urllib.parse.urlencode({'smiles': smiles}).encode('utf-8')

    # Make the request
    request = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode('utf-8'))

    # Extract base64 fingerprint from response
    fingerprint_b64 = result['PropertyTable']['Properties'][0]['Fingerprint2D']

    # Decode: base64 → bytes → binary string → numpy array
    fp_bytes = base64.b64decode(fingerprint_b64)
    bits = ''.join(format(b, '08b') for b in fp_bytes[4:])  # skip 4-byte header

    return np.array([int(b) for b in bits[:881]], dtype=np.int8)


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
    This class requires RDKit to be installed.
    Internet access is required to query the PubChem REST API.

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
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

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
            1D array of PubChem fingerprint bits. The length is 881.

        """
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        smiles = Chem.MolToSmiles(datapoint)
        return _get_pubchem_fingerprint(smiles)
