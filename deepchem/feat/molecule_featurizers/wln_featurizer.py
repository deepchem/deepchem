import numpy as np
import logging
from typing import List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond

from deepchem.feat.base_classes import MolecularFeaturizer

logger = logging.getLogger(__name__)

class PaddedGraphFeatures:
    """
    A data class to store padded graph features and labels for the Weisfeiler-Lehman Network (WLN).

    This class encapsulates the batched tensor representations of molecular graphs,
    including adjacency matrices, node features, and edge features, padded to the
    size of the largest molecule in the batch.

    Parameters
    ----------
    atom_features: np.ndarray
        A 3D array of shape (batch_size, max_atoms, atom_fdim) containing atom-level features.
    adjacency_matrix: np.ndarray
        A 3D array of shape (batch_size, max_atoms, max_atoms) representing the connectivity.
    bond_features: np.ndarray
        A 4D array of shape (batch_size, max_atoms, max_atoms, bond_fdim) containing bond-level features.
    binary_features: np.ndarray
        A 4D array of shape (batch_size, max_atoms, max_atoms, binary_fdim) containing binary pairwise features
        (e.g., component membership, bond existence) used for global attention.
    atom_masking: np.ndarray
        A 2D array of shape (batch_size, max_atoms) used to mask padding atoms.
    labels: np.ndarray
        A 4D array of shape (batch_size, max_atoms, max_atoms, nbos) containing the reaction edits
        (changes in bond order) as one-hot encoded labels.
    """

    def __init__(self, atom_features: np.ndarray, adjacency_matrix: np.ndarray,
                 bond_features: np.ndarray, binary_features: np.ndarray,
                 atom_masking: np.ndarray, labels: np.ndarray):
        self.atom_features = atom_features
        self.adjacency_matrix = adjacency_matrix
        self.bond_features = bond_features
        self.binary_features = binary_features
        self.atom_masking = atom_masking
        self.labels = labels


class WeisfeilerLehmanScoringModelFeaturizer(MolecularFeaturizer):
    """
    Featurizes molecules into padded graph representations for the Weisfeiler-Lehman Network (WLN).

    This featurizer is designed to predict organic reaction outcomes by modeling
    the "reaction center"—the specific atoms and bonds that change during a reaction.
    It processes reaction SMILES (e.g., "reactant1.reactant2>>product; edits") and
    extracts graph features for the **reactants**, relying on atom-mapping numbers
    to track atoms.

    The featurization process involves:
    1. Parsing reactant SMILES and edit strings.
    2. Constructing molecular graphs based on `molAtomMapNumber`.
    3. Generating local atom features (element, degree, valence, aromaticity).
    4. Generating pairwise features (bond types, conjugation, ring status).
    5. Generating global binary features for attention mechanisms (component membership).
    6. Padding all graphs in the batch to the size of the largest graph.

    The default node (atom) representation includes:
    - Atom type (One-hot)
    - Degree (One-hot)
    - Explicit Valence (One-hot)
    - Implicit Valence (One-hot)
    - Aromaticity (Boolean)

    The default edge (bond) representation includes:
    - Bond type (Single, Double, Triple, Aromatic)
    - Conjugation (Boolean)
    - Ring membership (Boolean)

    Parameters
    ----------
    max_atoms : int, optional (default None)
        The maximum number of atoms to pad to. If None, it is calculated dynamically
        based on the largest molecule in the batch.

    References
    ----------
    .. [1] Jin, Wengong, et al. "Predicting organic reaction outcomes with weisfeiler-lehman network."
           Advances in neural information processing systems 30 (2017). 
    .. [2] Coley, Connor W., et al. "A graph-convolutional neural network model for the prediction of chemical reactivity."
           Chemical Science 10.2 (2019).
    .. [3] https://github.com/connorcoley/rexgen_direct

    Note
    ----
    This class requires RDKit to be installed. All input SMILES must have 'molAtomMapNumber' properties.
    """

    def __init__(self):
        super(WeisfeilerLehmanScoringModelFeaturizer, self).__init__()

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
        """Helper function for one-hot encoding with fallback."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return np.array([x == s for s in allowable_set], dtype=np.float32)

    def _get_atom_features(self, atom: Atom) -> np.ndarray:
        """Construct an atom feature from a RDKit atom object."""
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
        """Construct a bond feature from a RDKit bond object."""
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
        Converts a single SMILES string to its graph representation using map numbers.

        Parameters
        ----------
        smiles_string: str
            The SMILES string to convert.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (atom_features, adj_matrix, bond_feature_matrix)
        """
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None, None, None

        num_atoms = mol.GetNumAtoms()

        try:
            sample_fdim = self._get_atom_features(mol.GetAtomWithIdx(0)).shape[0]
        except Exception:
            logger.warning(
                f"Could not get atom features for SMILES: {smiles_string}")
            return None, None, None

        atom_features = np.zeros((num_atoms, sample_fdim), dtype=np.float32)
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        bond_feature_matrix = np.zeros((num_atoms, num_atoms, self.bond_fdim),
                                       dtype=np.float32)

        for atom in mol.GetAtoms():
            try:
                map_num_idx = atom.GetIntProp('molAtomMapNumber') - 1
                if map_num_idx < 0 or map_num_idx >= num_atoms:
                    logger.warning(f"molAtomMapNumber out of bounds in {smiles_string}")
                    return None, None, None
                atom_features[map_num_idx] = self._get_atom_features(atom)
            except KeyError:
                return None, None, None

        for bond in mol.GetBonds():
            try:
                i = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
                j = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
                adj_matrix[i, j] = adj_matrix[j, i] = 1.0
                bond_feats = self._get_bond_features(bond)
                bond_feature_matrix[i, j] = bond_feature_matrix[j, i] = bond_feats
            except KeyError:
                return None, None, None

        return atom_features, adj_matrix, bond_feature_matrix

    def _binary_features(self, r_smiles: str, max_natoms: int) -> np.ndarray:
        """Generates binary features for global attention mechanisms."""
        mol = Chem.MolFromSmiles(r_smiles)
        features = np.zeros((max_natoms, max_natoms, self.binary_fdim),
                            dtype=np.float32)
        if mol is None:
            return features

        n_atoms = mol.GetNumAtoms()
        comp = {}
        try:
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
                    comp[idx_to_map_idx[atom_idx]] = i
        except (KeyError, ValueError):
            return features

        bond_map = {}
        for bond in mol.GetBonds():
            try:
                a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
                a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
                bond_map[(a1, a2)] = bond_map[(a2, a1)] = bond
            except KeyError:
                pass

        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue

                if (i, j) in bond_map:
                    features[i, j, 1:1 + self.bond_fdim] = self._get_bond_features(bond_map[(i, j)])
                else:
                    features[i, j, 0] = 1.0

                offset = 1 + self.bond_fdim
                features[i, j, offset] = 1.0 if comp[i] != comp[j] else 0.0
                features[i, j, offset + 1] = 1.0 if comp[i] == comp[j] else 0.0
                features[i, j, offset + 2] = 1.0 if n_comp == 1 else 0.0
                features[i, j, offset + 3] = 1.0 if n_comp > 1 else 0.0

        return features

    def _get_labels(self, edits: str, max_natoms: int) -> np.ndarray:
        """Parses the edit string from the datapoint to generate labels."""
        bo_to_index = {0.0: 0, 1: 1, 2: 2, 3: 3, 1.5: 4}
        nbos = len(bo_to_index)
        bond_map = np.zeros((max_natoms, max_natoms, nbos), dtype=np.float32)
        
        if not edits or edits.strip() == "":
            return bond_map

        for s in edits.split(';'):
            try:
                a1, a2, bo = s.split('-')
                x = int(a1) - 1
                y = int(a2) - 1
                z = bo_to_index[float(bo)]
                if x < max_natoms and y < max_natoms:
                    bond_map[x, y, z] = bond_map[y, x, z] = 1
            except (ValueError, IndexError):
                logger.warning(f"Could not parse edit string component: {s}")
                continue
        return bond_map

    def _featurize(self, datapoints: List[str], **kwargs) -> PaddedGraphFeatures:
        """
        Internal featurization logic. 
        
        This overrides the standard _featurize to return a single PaddedGraphFeatures
        object representing the entire batch.
        """
        graphs = []
        reactant_smiles = []
        edit_strings = []
        for s in datapoints:
            parts = s.split(">>")
            reactant_s = parts[0]
            reactant_smiles.append(reactant_s)
            graphs.append(self._smiles_to_graph(reactant_s))
            if len(parts) > 1 and " " in parts[1]:
                edit_s = parts[1].split(" ", 1)[-1]
                edit_strings.append(edit_s)
            else:
                edit_strings.append(None)

        valid_indices = [i for i, g in enumerate(graphs) if g[0] is not None]
        batch_size = len(datapoints)

        if not valid_indices:
            # Handle empty/invalid batch case
            atom_fdim = len(self.elem_list) + len(self.atom_degree_list) + \
                        len(self.atom_explicit_valence_list) + \
                        len(self.atom_implicit_valence_list) + 1
            max_atoms = 1
            nbos = 5
            return PaddedGraphFeatures(
                np.zeros((batch_size, max_atoms, atom_fdim), dtype=np.float32),
                np.zeros((batch_size, max_atoms, max_atoms), dtype=np.float32),
                np.zeros((batch_size, max_atoms, max_atoms, self.bond_fdim), dtype=np.float32),
                np.zeros((batch_size, max_atoms, max_atoms, self.binary_fdim), dtype=np.float32),
                np.zeros((batch_size, max_atoms), dtype=np.float32),
                np.zeros((batch_size, max_atoms, max_atoms, nbos), dtype=np.float32)
            )

        first_valid_graph = graphs[valid_indices[0]]
        atom_fdim = first_valid_graph[0].shape[1]
        bond_fdim = first_valid_graph[2].shape[2]
        max_atoms = max(graphs[i][1].shape[0] for i in valid_indices) + 1

        padded_atom_features = np.zeros((batch_size, max_atoms, atom_fdim), dtype=np.float32)
        padded_adj = np.zeros((batch_size, max_atoms, max_atoms), dtype=np.float32)
        padded_bond_features = np.zeros((batch_size, max_atoms, max_atoms, bond_fdim), dtype=np.float32)
        padded_binary_features = np.zeros((batch_size, max_atoms, max_atoms, self.binary_fdim), dtype=np.float32)
        atom_mask = np.zeros((batch_size, max_atoms), dtype=np.float32)
        labels = np.zeros((batch_size, max_atoms, max_atoms, 5), dtype=np.float32)

        for i in range(batch_size):
            atom_feats, adj, bond_feats = graphs[i]
            edit_s = edit_strings[i]

            if atom_feats is None:
                continue

            num_atoms = atom_feats.shape[0]
            padded_atom_features[i, :num_atoms] = atom_feats
            padded_adj[i, :num_atoms, :num_atoms] = adj
            padded_bond_features[i, :num_atoms, :num_atoms] = bond_feats
            atom_mask[i, :num_atoms] = 1.0

            reactant_s = reactant_smiles[i]
            binary_feats = self._binary_features(reactant_s, max_atoms)
            padded_binary_features[i] = binary_feats

            if edit_s is not None:
                labels[i] = self._get_labels(edit_s, max_natoms=max_atoms)

        return PaddedGraphFeatures(
            padded_atom_features,
            padded_adj,
            padded_bond_features,
            padded_binary_features,
            atom_mask,
            labels
        )
        
    def featurize(self,datapoints):
        """
        Featurizes a list of SMILES strings into padded batch representations for WLN.

        DATPOINTS FORMAT:
        REACTANT>>PRODUCT REACTION_CENTER EDITS

        EXAMPLES:
        [CH2:1]1[O:2][CH2:3][CH2:4][CH2:5]1.[CH:18]([Cl:19])([Cl:20])[Cl:21].[I:6][c:7]1[n:8][cH:9][cH:10][n:11][c:12]1[O:13][CH3:14].[NH2:16][NH2:17].[OH2:15]>>[c:7]1([NH:16][NH2:17])[n:8][cH:9][cH:10][n:11][c:12]1[O:13][CH3:14] 6-7-0.0;16-7-1.0

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