import logging
from typing import List

import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.feat.base_classes import Featurizer
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

ZINC_CHARSET = [
    '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=',
    '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o',
    'n', 'p', 's', 'r'
]


class OneHotFeaturizer(Featurizer):
    """Encodes any arbitrary string or molecule as a one-hot array.

    This featurizer encodes the characters within any given string as a one-hot
    array. It also works with RDKit molecules: it can convert RDKit molecules to
    SMILES strings and then one-hot encode the characters in said strings.

    Standalone Usage:

    >>> import deepchem as dc
    >>> featurizer = dc.feat.OneHotFeaturizer()
    >>> smiles = ['CCC']
    >>> encodings = featurizer.featurize(smiles)
    >>> type(encodings[0])
    <class 'numpy.ndarray'>
    >>> encodings[0].shape
    (100, 35)
    >>> featurizer.untransform(encodings[0])
    'CCC'

    Note
    ----
    This class needs RDKit to be installed in order to accept RDKit molecules as
    inputs.

    It does not need RDKit to be installed to work with arbitrary strings.

    """

    def __init__(self,
                 charset: List[str] = ZINC_CHARSET,
                 max_length: Optional[int] = 100):
        """Initialize featurizer.

        Parameters
        ----------
        charset: List[str] (default ZINC_CHARSET)
            A list of strings, where each string is length 1 and unique.
        max_length: Optional[int], optional (default 100)
            The max length for string. If the length of string is shorter than
            max_length, the string is padded using space.

        If max_length is None, no padding is performed and arbitrary length
        strings are allowed.
        """
        if len(charset) != len(set(charset)):
            raise ValueError("All values in charset must be unique.")
        self.charset = charset
        self.max_length = Optional[int]
        if max_length is not None:
            self.max_length = int(max_length)
        else:
            self.max_length = None

    def featurize(self,
                  datapoints: Iterable[Any],
                  log_every_n: int = 1000,
                  **kwargs) -> np.ndarray:
        """Featurize strings or mols.

        Parameters
        ----------
        datapoints: list
            A list of either strings (str or numpy.str_) or RDKit molecules.
        log_every_n: int, optional (default 1000)
            How many elements are featurized every time a featurization is logged.

        """
        datapoints = list(datapoints)
        if (len(datapoints) < 1):
            return np.array([])
        # Featurize data using featurize() in parent class
        return Featurizer.featurize(self, datapoints, log_every_n)

    def _featurize(self, datapoint: Any, **kwargs):
        # Featurize str data
        if isinstance(datapoint, (str, np.str_)):
            return self._featurize_string(datapoint)
        # Featurize mol data
        else:
            return self._featurize_mol(datapoint)

    def _featurize_string(self, string: str) -> np.ndarray:
        """Compute one-hot featurization of string.

        Parameters
        ----------
        string: str
            An arbitrary string to be featurized.

        Returns
        -------
        np.ndarray
            An one hot vector encoded from arbitrary input string.
            The shape is `(max_length, len(charset) + 1)`.
            The index of unknown character is `len(charset)`.

        """
        if isinstance(self.max_length, int):
            if (len(string) > self.max_length):  # Validation
                raise ValueError(
                    "The length of {} is longer than `max_length`.")
            string = self.pad_string(string)  # Padding

        return np.array([
            one_hot_encode(val, self.charset, include_unknown_set=True)
            for val in string
        ])

    def _featurize_mol(self, mol: RDKitMol) -> np.ndarray:
        """Compute one-hot featurization of this molecule.

        Parameters
        ----------
        mol: rdKit.Chem.rdchem.Mol
            RDKit Mol object

        Returns
        -------
        np.ndarray
            An one hot vector encoded from SMILES.
            The shape is '(max_length, len(charset) + 1)'
            The index of unknown character is 'len(charset)'.

        """
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")
        smiles = Chem.MolToSmiles(mol)  # Convert mol to SMILES string.
        return self._featurize_string(smiles)  # Use string featurization.

    def pad_smile(self, smiles: str) -> str:
        """Pad SMILES string to `self.pad_length`

        Parameters
        ----------
        smiles: str
            The SMILES string to be padded.

        Returns
        -------
        str
            SMILES string space padded to self.pad_length

        """
        return self.pad_string(smiles)

    def pad_string(self, string: str) -> str:
        """Pad string to `self.pad_length`

        Parameters
        ----------
        string: str
            The string to be padded.

        Returns
        -------
        str
            String space padded to self.pad_length

        """
        if isinstance(self.max_length, int):
            return string.ljust(self.max_length)
        else:
            return string

    def untransform(self, one_hot_vectors: np.ndarray) -> str:
        """Convert from one hot representation back to original string

        Parameters
        ----------
        one_hot_vectors: np.ndarray
            An array of one hot encoded features.

        Returns
        -------
        str
            Original string for an one hot encoded array.

        """
        string = ""
        for one_hot in one_hot_vectors:
            try:
                idx = np.argmax(one_hot)
                string += self.charset[idx]
            except IndexError:
                string += ""
        return string
