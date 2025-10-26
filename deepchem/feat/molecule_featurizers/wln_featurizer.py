import numpy as np
from typing import List, Optional, Tuple, Any, Dict
import torch
from deepchem.feat.base_classes import MolecularFeaturizer
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond
import logging

logger = logging.getLogger(__name__)


class WLNFeaturizer(MolecularFeaturizer):
    """
    Featurizes molecules into graph representations for use in graph neural networks.

    This featurizer is designed to parse SMILES strings, particularly
    reaction SMILES (e.g., "reactant1.reactant2>>product"), and extract
    graph features for the **reactants only**.

    It generates padded batches of:
    - Atom features
    - Adjacency matrices
    - Bond features
    - Binary features for global attention (e.g., bond/no-bond, same/different component)
    - Atom masks for padding

    **Crucially, this featurizer assumes all atoms in the input SMILES
    have a 'molAtomMapNumber' property,** which is typical for reaction
    SMILES. The graph indices are based on `molAtomMapNumber - 1`.

    Atom-level features include:
    - One-hot encoding of element type
    - One-hot encoding of atom degree
    - One-hot encoding of explicit valence
    - One-hot encoding of implicit valence
    - Aromaticity (boolean)

    Bond-level features include:
    - One-hot encoding of bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    - Conjugation (boolean)
    - Ring membership (boolean)
    """

    def __init__(self) -> None:
        """
        Initializes the featurizer with predefined feature sets.
        """
        super(WLNFeaturizer, self).__init__()
        # self.max_atoms is unused, padding is dynamic per batch
        # self.max_atoms = 50
        self.elem_list = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
            'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
            'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb',
            'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os',
            'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown'
        ]
        self.atom_degree_list = [0, 1, 2, 3, 4, 5]
        self.atom_explicit_valence_list = [1, 2, 3, 4, 5, 6]
        self.atom_implicit_valence_list = [0, 1, 2, 3, 4, 5]

        # Dimension of bond features from _get_bond_features
        self.bond_fdim = 6
        # Dimension of binary features from _binary_features
        # (1 no-bond + 6 bond_fdim + 4 component_feats) = 11
        self.binary_fdim = 11

    def _onek_encoding_unk(self, x: Any, allowable_set: List[Any]) -> np.ndarray:
        """
        Helper function for one-hot encoding with a fallback for unknown values.

        If `x` is not in `allowable_set`, it is mapped to the last element
        of the set (assumed to be 'unknown').

        Parameters
        ----------
        x: Any
            The value to be one-hot encoded.
        allowable_set: List[Any]
            A list of possible values.

        Returns
        -------
        np.ndarray
            A 1D numpy array of dtype float32.
        """
        if x not in allowable_set:
            x = allowable_set[-1]
        return np.array([x == s for s in allowable_set], dtype=np.float32)

    def _get_atom_features(self, atom: Atom) -> np.ndarray:
        """
        Calculates features for a single RDKit Atom.

        Parameters
        ----------
        atom: rdkit.Chem.rdchem.Atom
            The RDKit Atom object.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the concatenated atom features.
        """
        return np.concatenate([
            self._onek_encoding_unk(atom.GetSymbol(), self.elem_list),
            self._onek_encoding_unk(atom.GetDegree(), self.atom_degree_list),
            self._onek_encoding_unk(atom.GetExplicitValence(),
                                    self.atom_explicit_valence_list),
            self._onek_encoding_unk(atom.GetImplicitValence(),
                                    self.atom_implicit_valence_list),
            np.array([float(atom.GetIsAromatic())])
        ])

    def _get_bond_features(self, bond: Bond) -> np.ndarray:
        """
        Calculates features for a single RDKit Bond.

        Parameters
        ----------
        bond: rdkit.Chem.rdchem.Bond
            The RDKit Bond object.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the concatenated bond features.
        """
        bt = bond.GetBondType()
        features = np.array(
            [
                float(bt == Chem.rdchem.BondType.SINGLE),
                float(bt == Chem.rdchem.BondType.DOUBLE),
                float(bt == Chem.rdchem.BondType.TRIPLE),
                float(bt == Chem.rdchem.BondType.AROMATIC),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing())
            ],
            dtype=np.float32)
        return features

    def _smiles_to_graph(
        self, smiles_string: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Converts a single SMILES string to its graph representation.

        This function handles SMILES with '.' to represent multiple
        disconnected molecules. It relies on 'molAtomMapNumber' for
        indexing.

        Parameters
        ----------
        smiles_string: str
            The SMILES string to convert.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            A tuple containing (atom_features, adj_matrix, bond_feature_matrix).
            Returns (None, None, None) if the SMILES string is invalid or
            is missing 'molAtomMapNumber'.
        """
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None, None, None

        num_atoms = mol.GetNumAtoms()

        # Get feature dim from a sample atom
        try:
            sample_fdim = self._get_atom_features(mol.GetAtomWithIdx(0)).shape[0]
        except Exception:
            logger.warning(
                f"Could not get atom features for SMILES: {smiles_string}")
            return None, None, None

        # Initialize empty feature arrays based on num_atoms
        atom_features = np.zeros((num_atoms, sample_fdim), dtype=np.float32)
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        bond_feature_matrix = np.zeros((num_atoms, num_atoms, self.bond_fdim),
                                       dtype=np.float32)

        # Iterate over atoms to fill atom_features based on map number
        for atom in mol.GetAtoms():
            try:
                map_num_idx = atom.GetIntProp('molAtomMapNumber') - 1
                if map_num_idx < 0 or map_num_idx >= num_atoms:
                    raise ValueError("molAtomMapNumber out of bounds")
                atom_features[map_num_idx] = self._get_atom_features(atom)
            except (KeyError, ValueError):
                logger.warning(
                    f"Atom missing or has invalid 'molAtomMapNumber' in SMILES: {smiles_string}. Skipping molecule."
                )
                return None, None, None

        # Iterate over bonds to fill adj and bond_features based on map number
        for bond in mol.GetBonds():
            try:
                # Use map numbers for indexing
                i = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
                j = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
            except KeyError:
                logger.warning(
                    f"Bond atoms missing 'molAtomMapNumber' in SMILES: {smiles_string}. Skipping molecule."
                )
                return None, None, None

            adj_matrix[i, j] = adj_matrix[j, i] = 1.0
            bond_feats = self._get_bond_features(bond)
            bond_feature_matrix[i, j] = bond_feature_matrix[j, i] = bond_feats

        return atom_features, adj_matrix, bond_feature_matrix

    def _binary_features(self, r_smiles: str,
                         max_natoms: int) -> np.ndarray:
        """
        Generates binary features for global attention mechanisms.

        These features describe the relationship between any pair of atoms (i, j),
        indexed by their `molAtomMapNumber - 1`.

        Parameters
        ----------
        r_smiles: str
            The reactant SMILES string (can contain multiple components
            separated by '.').
        max_natoms: int
            The maximum number of atoms to pad to.

        Returns
        -------
        np.ndarray
            A 3D numpy array of shape (max_natoms, max_natoms, self.binary_fdim)
            containing the binary features.
        """
        mol = Chem.MolFromSmiles(r_smiles)
        # Initialize to zeros, which is correct for invalid SMILES or padding
        features = np.zeros((max_natoms, max_natoms, self.binary_fdim),
                            dtype=np.float32)
        if mol is None:
            return features

        n_atoms = mol.GetNumAtoms()

        # Get component membership for each atom *map number*
        comp = {}  # This will be indexed by map_idx (0 to N-1)
        try:
            # GetMolFrags returns tuples of *atom indices (GetIdx)*.
            # We must map GetIdx -> molAtomMapNumber - 1
            idx_to_map_idx = {
                atom.GetIdx(): atom.GetIntProp('molAtomMapNumber') - 1
                for atom in mol.GetAtoms()
            }
            if len(idx_to_map_idx) != n_atoms:
                raise ValueError("Duplicate or missing map numbers")

            frags_by_idx = Chem.GetMolFrags(mol, asMols=False)
            n_comp = len(frags_by_idx)

            for i, frag in enumerate(frags_by_idx):
                for atom_idx in frag:
                    map_idx = idx_to_map_idx[atom_idx]
                    comp[map_idx] = i  # comp is now indexed by map_idx

        except (KeyError, ValueError) as e:
            logger.warning(
                f"Error processing component fragments for SMILES: {r_smiles}. 'molAtomMapNumber' may be missing or invalid. Error: {e}"
            )
            # Return empty features, as we can't proceed
            return features

        # Get bond map using *map numbers*
        bond_map = {}  # This will be indexed by (map_idx_a, map_idx_b)
        try:
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
                a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
                bond_map[(a1, a2)] = bond_map[(a2, a1)] = bond
        except KeyError:
            logger.warning(
                f"Error processing bonds for SMILES: {r_smiles}. 'molAtomMapNumber' may be missing."
            )
            return features

        for i in range(n_atoms):  # i is a map_idx
            for j in range(n_atoms):  # j is a map_idx
                if i == j:
                    continue

                # f[0]: No Bond
                # f[1:7]: Bond features (if bond exists)
                if (i, j) in bond_map:
                    bond = bond_map[(i, j)]
                    features[i, j, 1:1 +
                                   self.bond_fdim] = self._get_bond_features(bond)
                else:
                    features[i, j, 0] = 1.0  # No bond

                # Features are offset by 1 (for no-bond) + 6 (for bond_fdim) = 7
                offset = 1 + self.bond_fdim
                # f[7]: Different component
                # f[8]: Same component
                features[i, j, offset] = 1.0 if comp[i] != comp[j] else 0.0
                features[i, j, offset + 1] = 1.0 if comp[i] == comp[j] else 0.0
                # f[9]: n_comp == 1
                # f[10]: n_comp > 1
                features[i, j, offset + 2] = 1.0 if n_comp == 1 else 0.0
                features[i, j, offset + 3] = 1.0 if n_comp > 1 else 0.0

        return features

    def _get_labels(self, edits):
        """
        (Placeholder) Parses the edit string from the datapoint.

        Parameters
        ----------
        edits: str
            The edit string, e.g., "2;;10-9-1.0;2-9-1.0"
        """
        pass

    def _featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal featurization function.

        Takes a list of SMILES strings (datapoints) and converts them
        into padded numpy arrays.

        Parameters
        ----------
        smiles_list: List[str]
            A list of SMILES strings. These can be reaction SMILES ("A.B>>C"),
            as only the reactant part ("A.B") will be featurized.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple of (padded_atom_features, padded_adj,
            padded_bond_features, padded_binary_features, atom_mask).
        """
        # 1. Parse SMILES and get graphs. Handle '>>' reaction syntax.
        graphs = []
        reactant_smiles = []
        edit_strings = []  # Placeholder for edit parsing

        for s in smiles_list:
            parts = s.split(">>")
            reactant_s = parts[0]
            reactant_smiles.append(reactant_s)
            graphs.append(self._smiles_to_graph(reactant_s))
            # Store edit strings if they exist
            if len(parts) > 1 and " " in parts[1]:
                edit_s = parts[1].split(" ", 1)[-1]
                edit_strings.append(edit_s)
            else:
                edit_strings.append(None)

        # 2. Find valid graphs to determine feature dimensions and max_atoms
        valid_indices = [i for i, g in enumerate(graphs) if g[0] is not None]
        batch_size = len(smiles_list)

        # 3. Handle edge case where ALL SMILES in the batch are invalid
        if not valid_indices:
            logger.warning("All SMILES in the batch are invalid.")
            # Get default feature dimensions
            atom_fdim = (
                len(self.elem_list) + len(self.atom_degree_list) +
                len(self.atom_explicit_valence_list) +
                len(self.atom_implicit_valence_list) + 1)
            bond_fdim = self.bond_fdim
            binary_fdim = self.binary_fdim
            max_atoms = 1  # Smallest possible padded size

            # Return all-zero tensors
            padded_atom_features = np.zeros((batch_size, max_atoms, atom_fdim),
                                            dtype=np.float32)
            padded_adj = np.zeros((batch_size, max_atoms, max_atoms),
                                  dtype=np.float32)
            padded_bond_features = np.zeros(
                (batch_size, max_atoms, max_atoms, bond_fdim),
                dtype=np.float32)
            padded_binary_features = np.zeros(
                (batch_size, max_atoms, max_atoms, binary_fdim),
                dtype=np.float32)
            atom_mask = np.zeros((batch_size, max_atoms), dtype=np.float32)

            return (padded_atom_features, padded_adj, padded_bond_features,
                    padded_binary_features, atom_mask)

        # 4. Get dimensions from the first valid graph
        first_valid_graph = graphs[valid_indices[0]]
        atom_fdim = first_valid_graph[0].shape[1]
        bond_fdim = first_valid_graph[2].shape[2]

        # 5. Determine max_atoms for THIS batch
        max_atoms = max(graphs[i][0].shape[0] for i in valid_indices)

        # 6. Initialize all padded arrays
        padded_atom_features = np.zeros((batch_size, max_atoms, atom_fdim),
                                        dtype=np.float32)
        padded_adj = np.zeros((batch_size, max_atoms, max_atoms),
                              dtype=np.float32)
        padded_bond_features = np.zeros(
            (batch_size, max_atoms, max_atoms, bond_fdim), dtype=np.float32)
        padded_binary_features = np.zeros(
            (batch_size, max_atoms, max_atoms, self.binary_fdim),
            dtype=np.float32)
        atom_mask = np.zeros((batch_size, max_atoms), dtype=np.float32)

        # 7. Fill the padded arrays
        for i in range(batch_size):
            atom_feats, adj, bond_feats = graphs[i]

            # Check if this specific graph is valid
            if atom_feats is None:
                continue  # Leave this entry as all-zeros

            num_atoms = atom_feats.shape[0]
            padded_atom_features[i, :num_atoms] = atom_feats
            padded_adj[i, :num_atoms, :num_atoms] = adj
            padded_bond_features[i, :num_atoms, :num_atoms] = bond_feats
            atom_mask[i, :num_atoms] = 1.0

            # Safely call _binary_features for this valid reactant
            reactant_s = reactant_smiles[i]
            binary_feats = self._binary_features(reactant_s, max_atoms)
            padded_binary_features[i] = binary_feats

            # (Placeholder) Parse labels from edit string
            # edit_s = edit_strings[i]
            # if edit_s:
            #   labels = self._get_labels(edit_s)
            #   ... add to a padded_labels array ...

        return (padded_atom_features, padded_adj, padded_bond_features,
                padded_binary_features, atom_mask)

    def featurize(self,
                  datapoints,
                  **kwargs):
        """
        Featurizes a list of SMILES strings into padded batch representations for WLN.

        DATPOINTS FORMAT:
        REACTANT>>PRODUCT REACTION_CENTER;;EDITS

        EXAMPLES:
        [CH3:1][NH:2][NH2:3].[CH3:4][CH2:5][O:6][C:7](=[O:8])[CH2:9][Br:10]>>CCOC(=O)CN(C)N 2;;10-9-1.0;2-9-1.0

        This method processes a list of datapoints (SMILES strings) and
        returns a tuple of numpy arrays, where each array represents a
        padded batch of a specific graph feature.

        If a reaction SMILES (e.g., "A.B>>C") is provided, only the
        reactants ("A.B") are featurized.

        Parameters
        ----------
        datapoints: List[str]
            A list of SMILES strings to featurize. **All atoms in these
            SMILES must have a 'molAtomMapNumber' property.**
        **kwargs:
            Additional keyword arguments.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing five numpy arrays:
            - **padded_atom_features**: (batch_size, max_atoms, atom_fdim)
            - **padded_adj**: (batch_size, max_atoms, max_atoms)
            - **padded_bond_features**: (batch_size, max_atoms, max_atoms, bond_fdim)
            - **padded_binary_features**: (batch_size, max_atoms, max_atoms, binary_fdim)
            - **atom_mask**: (batch_size, max_atoms)

        Example
        -------
        >>> from deepchem.feat.molecule_featurizers import WLNFeaturizer
        >>> # Initialize the featurizer
        >>> featurizer = WLNFeaturizer()
        >>>
        >>> # Example SMILES list (single molecules and a reaction)
        >>> smiles_list = [
        ... "[CH3:18][CH2:19][O:20][C:21](=[O:22])[Cl:23].[CH3:1][O:2][c:3]1[cH:4][c:5]2[n:6][c:7]([NH2:8])[c:9]([O:10][CH3:11])[n:12][c:13]2[cH:14][c:15]1[O:16][CH3:17].[cH:24]1[cH:25][cH:26][n:27][cH:28][cH:29]1>>CCOC(=O)Nc1nc2cc(OC)c(OC)cc2nc1OC",
        ... ]
        >>>
        >>> # Featurize
        >>> padded_atom_features, padded_adj, padded_bond_features, padded_binary_features, atom_mask = featurizer.featurize(smiles_list)
        >>>
        >>> print(f"Batch size: {padded_atom_features.shape[0]}")
        Batch size: 1
        >>> print(f"Max atoms: {padded_atom_features.shape[1]}")
        Max atoms: 29
        >>> print(f"Atom feature dim: {padded_atom_features.shape[2]}")
        Atom feature dim: 79
        >>> print(f"Atom mask sum: {atom_mask.sum()}")
        Atom mask sum: 29.0
        """

        return self._featurize(datapoints)