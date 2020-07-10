import numpy as np
from deepchem.feat.base_classes import MolecularFeaturizer

zinc_charset = [
    ' ', '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8',
    '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l',
    'o', 'n', 'p', 's', 'r'
]


class OneHotFeaturizer(MolecularFeaturizer):
  """Encodes a molecule as a one-hot array.

  This featurizer takes a molecule and encodes its Smiles string as a one-hot
  array.

  Note
  ----
  This class requires RDKit to be installed. Note that this featurizer is not
  Thread Safe in initialization of charset
  """

  def __init__(self, charset=None, padlength=120):
    """Initialize featurizer.

    Parameters
    ----------
    charset: list of str, optional (default None)
      A list of strings, where each string is length 1.
    padlength: int, optional (default 120)
      length to pad the smile strings to.
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")
    self.charset = charset
    self.pad_length = padlength

  def _featurize(self, mol):
    """Compute one-hot featurization of this molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.

    Returns
    -------
    rval: np.ndarray
      Vector of RDKit descriptors for `mol`
    """
    from rdkit import Chem
    smiles = Chem.MolToSmiles(mol)
    if self.charset is None:
      self.charset = self._create_charset(smiles)
    return np.array([self.one_hot_encoded(smile) for smile in smiles])

  def one_hot_array(self, i):
    """Create a one hot array with bit i set to 1

    Parameters
    ----------
    i: int
      bit to set to 1

    Returns
    -------
    obj:`list` of obj:`int`
      length len(self.charset)
    """
    return [int(x) for x in [ix == i for ix in range(len(self.charset))]]

  def one_hot_index(self, c):
    """Compute one-hot index of charater.

    Parameters
    ----------
    c: char
      character whose index we want

    Returns
    -------
    index of c in self.charset
    """
    return self.charset.index(c)

  def pad_smile(self, smile):
    """Pad a smile string to `self.pad_length`

    Parameters
    ----------
    smile: str
      The smiles string to be padded.

    Returns
    -------
    str
      smile string space padded to self.pad_length
    """

    return smile.ljust(self.pad_length)

  def one_hot_encoded(self, smile):
    """One Hot Encode an entire SMILE string
    
    Parameters
    ----------
    smile: str
      smile string to encode

    Returns
    -------
    np.array of one hot encoded arrays for each character in smile
    """
    return np.array([
        self.one_hot_array(self.one_hot_index(x)) for x in self.pad_smile(smile)
    ])

  def untransform(self, z):
    """Convert from one hot representation back to SMILE

    Parameters
    ----------
    z: obj:`list`
      list of one hot encoded features

    Returns
    -------
    Smile Strings picking MAX for each one hot encoded array
    """
    z1 = []
    for i in range(len(z)):
      s = ""
      for j in range(len(z[i])):
        oh = np.argmax(z[i][j])
        s += self.charset[oh]
      z1.append([s.strip()])
    return z1

  def _create_charset(self, smiles):
    """Create the charset from smiles

    Parameters
    ----------
    smiles: obj:`list` of obj:`str`
      list of smile strings

    Returns
    -------
    obj:`list` of obj:`str`
      List of length one strings that are characters in smiles.  No duplicates
    """
    s = set()
    for smile in smiles:
      for c in smile:
        s.add(c)
    return [' '] + sorted(list(s))
