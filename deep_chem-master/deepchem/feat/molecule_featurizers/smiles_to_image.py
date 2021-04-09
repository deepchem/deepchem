"""
Featurizer implementations used in ChemCeption models.
SmilesToImage featurizer for ChemCeption models taken from https://arxiv.org/abs/1710.02238
"""
import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer


class SmilesToImage(MolecularFeaturizer):
  """Convert SMILES string to an image.

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
               img_size: int = 80,
               res: float = 0.5,
               max_len: int = 250,
               img_spec: str = "std"):
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

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """Featurizes a single SMILE into an image.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      A 3D array of image, the shape is `(img_size, img_size, 1)`.
      If the length of SMILES is longer than `max_len`, this value is an empty array.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import AllChem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    smile = Chem.MolToSmiles(mol)
    if len(smile) > self.max_len:
      return np.array([])

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
        return np.array([])

    frac = np.linspace(0, 1, int(1 / self.res * 2))
    # Reshape done for proper broadcast
    frac = frac.reshape(-1, 1, 1)

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
