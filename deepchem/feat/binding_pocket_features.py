"""
Featurizes proposed binding pockets.
"""
import numpy as np
import logging
from typing import Dict, List

from deepchem.feat import Featurizer
from deepchem.utils.coordinate_box_utils import CoordinateBox
from deepchem.utils.rdkit_utils import load_molecule

logger = logging.getLogger(__name__)


def boxes_to_atoms(
        coords: np.ndarray,
        boxes: List[CoordinateBox]) -> Dict[CoordinateBox, List[int]]:
    """Maps each box to a list of atoms in that box.

    Given the coordinates of a macromolecule, and a collection of boxes,
    returns a dictionary which maps boxes to the atom indices of the
    atoms in them.

    Parameters
    ----------
    coords: np.ndarray
        A numpy array of shape `(N, 3)`
    boxes: list
        List of `CoordinateBox` objects.

    Returns
    -------
    Dict[CoordinateBox, List[int]]
        A dictionary mapping `CoordinateBox` objects to lists of atom indices.
    """
    mapping = {}
    for box_ind, box in enumerate(boxes):
        box_atoms = []
        for atom_ind in range(len(coords)):
            atom = coords[atom_ind]
            if atom in box:
                box_atoms.append(atom_ind)
        mapping[box] = box_atoms
    return mapping


class BindingPocketFeaturizer(Featurizer):
    """Featurizes binding pockets with information about chemical
    environments.

    In many applications, it's desirable to look at binding pockets on
    macromolecules which may be good targets for potential ligands or
    other molecules to interact with. A `BindingPocketFeaturizer`
    expects to be given a macromolecule, and a list of pockets to
    featurize on that macromolecule. These pockets should be of the form
    produced by a `dc.dock.BindingPocketFinder`, that is as a list of
    `dc.utils.CoordinateBox` objects.

    The base featurization in this class's featurization is currently
    very simple and counts the number of residues of each type present
    in the pocket. It's likely that you'll want to overwrite this
    implementation for more sophisticated downstream usecases. Note that
    this class's implementation will only work for proteins and not for
    other macromolecules

    Note
    ----
    This class requires mdtraj to be installed.
    """

    residues = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "PYL", "SER", "SEC", "THR", "TRP",
        "TYR", "VAL", "ASX", "GLX"
    ]

    n_features = len(residues)

    # FIXME: Signature of "featurize" incompatible with supertype "Featurizer"
    def featurize(  # type: ignore[override]
            self, protein_file: str,
            pockets: List[CoordinateBox]) -> np.ndarray:
        """
        Calculate atomic coodinates.

        Parameters
        ----------
        protein_file: str
            Location of PDB file. Will be loaded by MDTraj
        pockets: List[CoordinateBox]
            List of `dc.utils.CoordinateBox` objects.

        Returns
        -------
        np.ndarray
            A numpy array of shale `(len(pockets), n_residues)`
        """
        try:
            import mdtraj
        except ModuleNotFoundError:
            raise ImportError("This class requires mdtraj to be installed.")

        protein_coords = load_molecule(protein_file,
                                       add_hydrogens=False,
                                       calc_charges=False)[0]
        mapping = boxes_to_atoms(protein_coords, pockets)
        protein = mdtraj.load(protein_file)
        n_pockets = len(pockets)
        n_residues = len(BindingPocketFeaturizer.residues)
        res_map = dict(zip(BindingPocketFeaturizer.residues, range(n_residues)))
        all_features = np.zeros((n_pockets, n_residues))
        for pocket_num, pocket in enumerate(pockets):
            pocket_atoms = mapping[pocket]
            for ind, atom in enumerate(pocket_atoms):
                atom_name = str(protein.top.atom(atom))
                # atom_name is of format RESX-ATOMTYPE
                # where X is a 1 to 4 digit number
                residue = atom_name[:3]
                if residue not in res_map:
                    logger.info("Warning: Non-standard residue in PDB file")
                    continue
                all_features[pocket_num, res_map[residue]] += 1
        return all_features
