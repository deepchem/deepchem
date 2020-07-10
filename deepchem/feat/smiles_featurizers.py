"""
Featurizer implementations used in ChemCeption and Smiles2Vec models.
SmilesToSeq featurizer for Smiles2Vec models taken from https://arxiv.org/abs/1712.02734
SmilesToImage featurizer for ChemCeption models taken from https://arxiv.org/abs/1710.02238
"""

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import pandas as pd
from deepchem.feat.base_classes import MolecularFeaturizer

PAD_TOKEN = "<pad>"
OUT_OF_VOCAB_TOKEN = "<unk>"


def create_char_to_idx(filename,
                       max_len=250,
                       smiles_field="smiles",
                       verbose=False):
  """Creates a dictionary with character to index mapping.

  Parameters
  ----------
  filename: str,
      Name of the file containing the SMILES strings
  max_len: int, default 250
      Maximum allowed length of the SMILES string
  smiles_field: str, default smiles
      Field indicating the SMILES strings int the file.
  verbose: bool, default True
      Whether to print the progress

  Returns
  -------
  A dictionary mapping characters to their integer indexes.
  """
  smiles_df = pd.read_csv(filename)
  char_set = set()
  for smile in smiles_df[smiles_field]:
    if len(smile) <= max_len:
      char_set.update(set(smile))

  unique_char_list = list(char_set)
  unique_char_list += [PAD_TOKEN, OUT_OF_VOCAB_TOKEN]
  if verbose:
    print("Number of unique characters: ", len(unique_char_list))

  char_to_idx = {letter: idx for idx, letter in enumerate(unique_char_list)}

  if verbose:
    print(unique_char_list)
  return char_to_idx


class SmilesToSeq(MolecularFeaturizer):
  """
  SmilesToSeq Featurizer takes a SMILES string, and turns it into a sequence.
  Details taken from [1]_.

  SMILES strings smaller than a specified max length (max_len) are padded using
  the PAD token while those larger than the max length are not considered. Based
  on the paper, there is also the option to add extra padding (pad_len) on both
  sides of the string after length normalization. Using a character to index (char_to_idx)
  mapping, the SMILES characters are turned into indices and the
  resulting sequence of indices serves as the input for an embedding layer.

  References
  ----------
  .. [1] Goh, Garrett B., et al. "Using rule-based labels for weak supervised
         learning: a ChemNet for transferable chemical property prediction."
         Proceedings of the 24th ACM SIGKDD International Conference on Knowledge
         Discovery & Data Mining. 2018.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, char_to_idx, max_len=250, pad_len=10, **kwargs):
    """Initialize this class. 

    Parameters
    ----------
    char_to_idx: dict
        Dictionary containing character to index mappings for unique characters
    max_len: int, default 250
        Maximum allowed length of the SMILES string
    pad_len: int, default 10
        Amount of padding to add on either side of the SMILES seq
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")
    self.max_len = max_len
    self.char_to_idx = char_to_idx
    self.idx_to_char = {idx: letter for letter, idx in self.char_to_idx.items()}
    self.pad_len = pad_len
    super(SmilesToSeq, self).__init__(**kwargs)

  def to_seq(self, smile):
    """Turns list of smiles characters into array of indices"""
    out_of_vocab_idx = self.char_to_idx[OUT_OF_VOCAB_TOKEN]
    seq = [
        self.char_to_idx.get(character, out_of_vocab_idx) for character in smile
    ]
    return np.array(seq)

  def remove_pad(self, characters):
    """Removes PAD_TOKEN from the character list."""
    characters = characters[self.pad_len:]
    characters = characters[:-self.pad_len]
    chars = list()

    for char in characters:
      if char != PAD_TOKEN:
        chars.append(char)
    return chars

  def smiles_from_seq(self, seq):
    """Reconstructs SMILES string from sequence."""
    characters = [self.idx_to_char[i] for i in seq]

    characters = self.remove_pad(characters)
    smile = "".join([letter for letter in characters])
    return smile

  def _featurize(self, mol):
    """Featurizes a SMILES sequence."""
    from rdkit import Chem
    smile = Chem.MolToSmiles(mol)
    if len(smile) > self.max_len:
      return list()

    smile_list = list(smile)
    # Extend shorter strings with padding
    if len(smile) < self.max_len:
      smile_list.extend([PAD_TOKEN] * (self.max_len - len(smile)))

    # Padding before and after
    smile_list += [PAD_TOKEN] * self.pad_len
    smile_list = [PAD_TOKEN] * self.pad_len + smile_list

    smile_seq = self.to_seq(smile_list)
    return smile_seq


class SmilesToImage(MolecularFeaturizer):
  """Convert Smiles string to an image.

  SmilesToImage Featurizer takes a SMILES string, and turns it into an image.
  Details taken from [1]_.

  The default size of for the image is 80 x 80. Two image modes are currently
  supported - std & engd. std is the gray scale specification,
  with atomic numbers as pixel values for atom positions and a constant value of
  2 for bond positions. engd is a 4-channel specification, which uses atom
  properties like hybridization, valency, charges in addition to atomic number.
  Bond type is also used for the bonds.

  The coordinates of all atoms are computed, and lines are drawn between atoms
  to indicate bonds. For the respective channels, the atom and bond positions are
  set to the property values as mentioned in the paper.

  References
  ----------
  .. [1] Goh, Garrett B., et al. "Using rule-based labels for weak supervised
         learning: a ChemNet for transferable chemical property prediction."
         Proceedings of the 24th ACM SIGKDD International Conference on Knowledge
         Discovery & Data Mining. 2018.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self,
               img_size=80,
               res=0.5,
               max_len=250,
               img_spec="std",
               **kwargs):
    """
    Parameters
    ----------
    img_size: int, default 80
        Size of the image tensor
    res: float, default 0.5
        Displays the resolution of each pixel in Angstrom
    max_len: int, default 250
        Maximum allowed length of SMILES string
    img_spec: str, default std
        Indicates the channel organization of the image tensor
    """
    if img_spec not in ["std", "engd"]:
      raise ValueError(
          "Image mode must be one of std or engd. {} is not supported".format(
              img_spec))
    self.img_size = img_size
    self.max_len = max_len
    self.res = res
    self.img_spec = img_spec
    self.embed = int(img_size * res / 2)
    super(SmilesToImage, self).__init__()

  def _featurize(self, mol):
    """Featurizes a single SMILE sequence."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smile = Chem.MolToSmiles(mol)
    if len(smile) > self.max_len:
      return list()

    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    atom_coords = cmol.GetConformer(0).GetPositions()

    if self.img_spec == "std":
      # Setup image
      img = np.zeros((self.img_size, self.img_size, 1))
      # Compute bond properties
      bond_props = np.array(
          [[2.0, bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
      # Compute atom properties
      atom_props = np.array([[atom.GetAtomicNum()] for atom in cmol.GetAtoms()])

      bond_props = bond_props.astype(np.float32)
      atom_props = atom_props.astype(np.float32)

    else:
      # Setup image
      img = np.zeros((self.img_size, self.img_size, 4))
      # Compute bond properties
      bond_props = np.array([[
          bond.GetBondTypeAsDouble(),
          bond.GetBeginAtomIdx(),
          bond.GetEndAtomIdx()
      ] for bond in mol.GetBonds()])
      # Compute atom properties
      atom_props = np.array([[
          atom.GetAtomicNum(),
          atom.GetProp("_GasteigerCharge"),
          atom.GetExplicitValence(),
          atom.GetHybridization().real,
      ] for atom in cmol.GetAtoms()])

      bond_props = bond_props.astype(np.float32)
      atom_props = atom_props.astype(np.float32)

      partial_charges = atom_props[:, 1]
      if np.any(np.isnan(partial_charges)):
        return []

    frac = np.linspace(0, 1, int(1 / self.res * 2))
    # Reshape done for proper broadcast
    frac = frac.reshape(-1, 1, 1)

    try:
      bond_begin_idxs = bond_props[:, 1].astype(int)
      bond_end_idxs = bond_props[:, 2].astype(int)

      # Reshapes, and axes manipulations to facilitate vector processing.
      begin_coords = atom_coords[bond_begin_idxs]
      begin_coords = np.expand_dims(begin_coords.T, axis=0)
      end_coords = atom_coords[bond_end_idxs]
      end_coords = np.expand_dims(end_coords.T, axis=0)

      # Draw a line between the two atoms.
      # The coordinates of this line, are indicated in line_coords
      line_coords = frac * begin_coords + (1 - frac) * end_coords
      # Turn the line coordinates into image positions
      bond_line_idxs = np.ceil(
          (line_coords[:, 0] + self.embed) / self.res).astype(int)
      bond_line_idys = np.ceil(
          (line_coords[:, 1] + self.embed) / self.res).astype(int)
      # Set the bond line coordinates to the bond property used.
      img[bond_line_idxs, bond_line_idys, 0] = bond_props[:, 0]

      # Turn atomic coordinates into image positions
      atom_idxs = np.round(
          (atom_coords[:, 0] + self.embed) / self.res).astype(int)
      atom_idys = np.round(
          (atom_coords[:, 1] + self.embed) / self.res).astype(int)
      # Set the atom positions in image to different atomic properties in channels
      img[atom_idxs, atom_idys, :] = atom_props
      return img

    except IndexError as e:
      return []
