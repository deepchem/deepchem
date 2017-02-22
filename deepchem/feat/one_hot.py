import numpy as np
from deepchem.feat import Featurizer
from rdkit import Chem

zinc_charset = [' ', '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=', '@', 'C', 'B', 'F', 'I',
                'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o', 'n', 'p', 's', 'r']

class OneHotFeaturizer(Featurizer):
  """
  NOTE(LESWING) Not Thread Safe in initialization of charset
  """

  def __init__(self, charset, padlength=120):
    self.charset = charset
    self.padlength = padlength

  def featurize(self, mols, verbose=True, log_every_n=1000):
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    if self.charset is None:
      self.charset = self._create_charset(mols)
    return np.array([self.one_hot_encoded(smile) for smile in smiles])

  def one_hot_array(self, i):
    return [int(x) for x in [ix == i for ix in range(len(self.charset))]]

  def one_hot_index(self, c):
    """
    TODO(LESWING) replace with map lookup vs linear scan
    :param charset:
    :return:
    """
    return self.charset.index(c)

  def pad_smile(self, smile):
    return smile.ljust(self.padlength)

  def one_hot_encoded(self, smile):
    return np.array([self.one_hot_array(self.one_hot_index(x)) for x in self.pad_smile(smile)])

  def untransform(self, z):
    z1 = []
    for i in range(len(z)):
      s = ""
      for j in range(len(z[i])):
        oh = np.argmax(z[i][j])
        s += self.charset[oh]
      z1.append([s.strip()])
    return z1

  def _create_charset(self, smiles):
    s = set()
    for smile in smiles:
      s.union(list(smile))
    return sorted(list(s))
