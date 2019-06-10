"""
Featurizer implementations used in ChemCeption and Smiles2Vec models.
"""

from __future__ import division
from __future__ import unicode_literals

__author__ = "Vignesh Ram Somnath"
__license__ = "MIT"

import numpy as np
import pandas as pd
from deepchem.feat import Featurizer

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


class SmilesToSeq(Featurizer):

  def __init__(self, char_to_idx, max_len=250, pad_len=10, **kwargs):
    """
    Parameters
    ----------
    char_to_idx: dict
        Dictionary containing character to index mappings for unique characters
    max_len: int, default 250
        Maximum allowed length of the SMILES string
    pad_len: int, default 10
        Amount of padding to add on either side of the SMILES seq
    """
    self.max_len = max_len
    self.char_to_idx = char_to_idx
    self.idx_to_char = {idx: letter for letter, idx in self.char_to_idx.items()}
    self.pad_len = pad_len
    super(SmilesToSeq, self).__init__(**kwargs)

  def to_seq(self, smile):
    """Turns list of smiles characters into array of indices"""
    seq = [self.char_to_idx[character] for character in smile]
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


class SmilesToImage(Featurizer):

  def __init__(self,
               img_size=80,
               res=0.5,
               embed=20,
               max_len=250,
               img_mode="std",
               **kwargs):
    """
    Parameters
    ----------
    img_size: int, default 80
        Size of the image tensor
    res: float, default 0.5
        Displays the resolution of each pixel in Angstrom
    embded: int, default 20
        #TODO
    max_len: int, default 250
        Maximum allowed length of SMILES string
    img_mode: str, default std
        Indicates the channel organization of the image tensor
    """
    if img_mode not in ["std", "engd"]:
      raise ValueError(
          "Image mode must be one of std or engd. {} is not supported".format(
              img_mode))
    self.img_size = img_size
    self.max_len = max_len
    self.res = res
    self.img_mode = img_mode
    self.embed = embed
    super(SmilesToImage, self).__init__(**kwargs)

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

    if self.img_mode == "std":
      # Setup image
      img = np.zeros((self.img_size, self.img_size, 1))
      # Compute bond properties
      bond_props = np.array(
          [[2.0, bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
      # Compute atom properties
      atom_props = np.array([[atom.GetAtomicNum()] for atom in cmol.GetAtoms()])

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
