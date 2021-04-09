"""
Atomic coordinate featurizer.
"""
import numpy as np

from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol


class AtomicCoordinates(MolecularFeaturizer):
  """Calculate atomic coordinates.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, use_bohr: bool = False):
    """
    Parameters
    ----------
    use_bohr: bool, optional (default False)
      Whether to use bohr or angstrom as a coordinate unit.
    """
    self.use_bohr = use_bohr

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """Calculate atomic coordinates.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      A numpy array of atomic coordinates. The shape is `(n_atoms, 3)`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import AllChem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    # Check whether num_confs >=1 or not
    num_confs = len(mol.GetConformers())
    if num_confs == 0:
      mol = Chem.AddHs(mol)
      AllChem.EmbedMolecule(mol, AllChem.ETKDG())
      mol = Chem.RemoveHs(mol)

    N = mol.GetNumAtoms()
    coords = np.zeros((N, 3))

    # RDKit stores atomic coordinates in Angstrom. Atomic unit of length is the
    # bohr (1 bohr = 0.529177 Angstrom). Converting units makes gradient calculation
    # consistent with most QM software packages.
    if self.use_bohr:
      coords_list = [
          mol.GetConformer(0).GetAtomPosition(i).__idiv__(0.52917721092)
          for i in range(N)
      ]
    else:
      coords_list = [mol.GetConformer(0).GetAtomPosition(i) for i in range(N)]

    for atom in range(N):
      coords[atom, 0] = coords_list[atom].x
      coords[atom, 1] = coords_list[atom].y
      coords[atom, 2] = coords_list[atom].z

    return coords
