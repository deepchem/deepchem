from rdkit import Chem
from collections import Counter
import numpy as np


class FeaturizationParameters:
    """
    Class for storing featurization parameters.
    This class holds all the parameter required to encode the feature vector
    both for atoms and bonds

    Attributes
    ----------
    MAX_ATOMIC_NUM : int
        Maximum atomic number in the dataset
    ATOM_FEATURES : dict
        Dictionary containing the atom features
    PATH_DISTANCE_BINS : list
        List containing the path distance bins
    THREE_D_DISTANCE_MAX : int
        Maximum 3D distance in the dataset
    THREE_D_DISTANCE_STEP : int
        Step size for 3D distance
    THREE_D_DISTANCE_BINS : list
        List containing the 3D distance bins
    ATOM_FDIM : int
        Atom feature dimension
    EXTRA_ATOM_FDIM : int
        Extra atom feature dimension
    BOND_FDIM : int
        Bond feature dimension
    EXTRA_BOND_FDIM : int
        Extra bond feature dimension
    REACTION_MODE : str
        Reaction mode
        Can be 'reac' or 'prod'
    EXPLICIT_H : bool
        Whether to use explicit hydrogen
    REACTION : bool
        Whether to use reaction mode
    POLYMER : bool
        To notify that the featurization is done for polymers
    ADDING_H : bool
        To notify that the hydrogen will be added in molecules while featurization
    """

    def __init__(self) -> None:
        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num':
                list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D,
            ]
        }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(
            range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(
            len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.POLYMER = False
        self.ADDING_H = False


def handle_hydrogen(smiles: str,
                    keep_h: bool = False,
                    add_h: bool = False) -> Chem.rdchem.Mol:
    """
    Builds an RDKit molecule from a SMILES string by conditionally
    handling addition or removal of hydrogens.

    This function is useful to process organic molecules properly.

    Parameters
    ----------
    smiles : str
        SMILES string.
    keep_h : bool, optional, default=False
        Whether to keep hydrogens in the molecule.
    add_h : bool, optional, default=False
        Whether to add hydrogens to the molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit molecule object.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol = dc.utils.handle_hydrogen('C', keep_h=True, add_h=False)
    >>> mol.GetNumAtoms()
    1
    >>> mol = dc.utils.handle_hydrogen('C', keep_h=True, add_h=True)
    >>> mol.GetNumAtoms()
    5
    """
    if keep_h:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    else:
        mol = Chem.MolFromSmiles(smiles)
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


def make_polymer_mol(
    smiles: str,
    fragment_weights: list[float],
    keep_h: bool,
    add_h: bool,
) -> Chem.rdchem.Mol:
    """
    Builds an RDKit joined molecule from a SMILES string of monomer molecules.
    The weight of each monomer is stored as a metadata property of each atom.

    Parameters
    ----------
    smiles : str
        SMILES string of monomer molecules joined by ".".
    fragment_weights : list
        List of weights for each monomer.
    keep_h : bool
        Whether to keep hydrogens in the molecule.
    add_h : bool
        Whether to add hydrogens to the molecule.
    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit polymer molecule object.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol = dc.utils.make_polymer_mol('C.C', True, False, [1, 1])
    >>> for atom in mol.GetAtoms():
    ...     print(atom.GetDoubleProp('w_frag'))
    1.0
    1.0
    """

    # check input is correct, we need the same number of fragments and their weights
    num_frags = len(smiles.split('.'))
    if len(fragment_weights) != num_frags:
        raise ValueError(
            f'number of input monomers/fragments ({num_frags}) does not match number of '
            f'input number of weights ({len(fragment_weights)})')

    # if it all looks good, we create one molecule object per fragment, add the weight as property
    # of each atom, and merge fragments into a single molecule object
    mols = []
    for s, w in zip(smiles.split('.'), fragment_weights):
        m = handle_hydrogen(s, keep_h, add_h)
        for a in m.GetAtoms():
            a.SetDoubleProp(
                'w_frag', float(w)
            )  # assinging metadata to store the weight fragment of the monomer
        mols.append(m)

    # combine all mols into single mol object
    mol = mols.pop(0)
    while len(mols) > 0:
        m2 = mols.pop(0)
        mol = Chem.CombineMols(mol, m2)

    return mol


def parse_polymer_rules(rules: list[str]) -> (list[tuple], float):
    """
    This function extracts probabilty weight distribution details for bonds
    from string to list of tuples in following format.
    (start, end, weight_forward, weight_reverse)
    The start and end is in string. (As it will be mapped with open bond string)
    The weight_forward and weight_reverse in float
    It also returns the degree of polymerization of the polymer (DoP) as 1. + np.log10(DoP).
    The DoP can be mentioned at the end of the rules separated by "~".
    If DoP is not given, the function assumes DoP = 1.

    Parameters
    ----------
    rules : list[str]
        List of strings containing bond rules in the format
        "start-end:weight_forward:weight_reverse"
    Returns:
    --------
    polymer_info : list[tuple]
        List of tuples containing bond rules in the format
        (start, end, weight_forward, weight_reverse)
    degree_of_polymerization : float
        Degree of polymerization of the polymer.

    Examples
    --------
    >>> import deepchem as dc
    >>> polymer_info, degree_of_polymerization = dc.utils.parse_polymer_rules(
    ...     ['1-2:0.5:0.5'])
    >>> polymer_info
    [('1', '2', 0.5, 0.5)]
    >>> degree_of_polymerization
    1.0
    """
    polymer_info = []
    counter = Counter()  # used for validating the input

    # check if deg of polymerization is provided
    if '~' in rules[-1]:
        Xn = float(rules[-1].split('~')[1])
        rules[-1] = rules[-1].split('~')[0]
    else:
        Xn = 1.

    for rule in rules:
        # handle edge case where we have no rules, and rule is empty string
        if rule == "":
            continue
        # QC of input string
        if len(rule.split(':')) != 3:
            raise ValueError(f'incorrect format for input information "{rule}"')
        idx1, idx2 = rule.split(':')[0].split('-')
        w12 = float(rule.split(':')[1])  # weight for bond R_idx1 -> R_idx2
        w21 = float(rule.split(':')[2])  # weight for bond R_idx2 -> R_idx1
        polymer_info.append((idx1, idx2, w12, w21))
        counter[idx1] += float(w21)
        counter[idx2] += float(w12)

    # validate input: sum of incoming weights should be one for each vertex
    for k, v in counter.items():
        if np.isclose(v, 1.0) is False:
            raise ValueError(
                f'sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]'
            )
    return polymer_info, 1. + np.log10(Xn)


def tag_atoms_in_repeating_unit(mol):
    """
    This function tags atoms that are part of the core units, as well as atoms
    serving to identify attachment points. In addition, create a map of bond
    types based on what bonds are connected to R groups in the input.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    dict
        Map of R group to bond type.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol, _ = dc.utils.tag_atoms_in_repeating_unit(Chem.MolFromSmiles('[1*]CC.C[2*]'))
    >>> mol.GetAtomWithIdx(0).GetBoolProp('core')
    False
    >>> mol.GetAtomWithIdx(1).GetBoolProp('core')
    True
    """
    atoms = [a for a in mol.GetAtoms()]
    neighbor_map = {}  # map R group to index of atom it is attached to
    r_bond_types = {}  # map R group to bond type

    # go through each atoms and: (i) get index of attachment atoms, (ii) tag all non-R atoms
    for atom in atoms:
        # if R atom
        if '*' in atom.GetSmarts():
            # get index of atom it is attached to
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1
            neighbor_idx = neighbors[0].GetIdx()
            r_tag = atom.GetSmarts().strip('[]').replace(':', '')  # *1, *2, ...
            neighbor_map[r_tag] = neighbor_idx
            # tag it as non-core atom
            atom.SetBoolProp('core', False)
            # create a map R --> bond type
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
            r_bond_types[r_tag] = bond.GetBondType()
        # if not R atom
        else:
            # tag it as core atom
            atom.SetBoolProp('core', True)

    # use the map created to tag attachment atoms
    for atom in atoms:
        if atom.GetIdx() in neighbor_map.values():
            r_tags = [k for k, v in neighbor_map.items() if v == atom.GetIdx()]
            atom.SetProp('R', ''.join(r_tags))
        else:
            atom.SetProp('R', '')

    return mol, r_bond_types


def onek_encoding_unk(value: int, choices: list) -> list:
    """
    This function generates the vector for a value as one hot encoded list.
    If there is an unknown value, it will be encoded as 1 in the last index

    Parameters
    ----------
    value : int
        Value to be encoded.
    choices : list
        List of choices.

    Returns
    -------
    list
        One hot encoded vector.
    Examples
    --------
    >>> import deepchem as dc
    >>> dc.utils.onek_encoding_unk(1, [1, 2, 3])
    [1, 0, 0, 0]
    >>> dc.utils.onek_encoding_unk(69, [1, 2, 3])
    [0, 0, 0, 1]
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def generate_atom_features(atom: Chem.rdchem.Atom,
                           PARAMS: FeaturizationParameters,
                           functional_groups=None):
    """
    This function generates the feature vector for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom object.
    PARAMS : deepchem.feat.FeaturizationParameters
        Featurization parameters.
    functional_groups : list, optional
        List of functional groups.

    Returns
    -------
    list
        Feature vector.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles("C")
    >>> PARAMS = dc.feat.FeaturizationParameters()
    >>> for atom in mol.GetAtoms():
    ...     atom_feat_vector = dc.utils.generate_atom_features(
    ...         atom, PARAMS = PARAMS))
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0.12011]
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum(
        ) - 1, PARAMS.ATOM_FEATURES['atomic_num']) + onek_encoding_unk(
            atom.GetTotalDegree(),
            PARAMS.ATOM_FEATURES['degree']) + onek_encoding_unk(
                atom.GetFormalCharge(),
                PARAMS.ATOM_FEATURES['formal_charge']) + onek_encoding_unk(
                    int(atom.GetChiralTag()),
                    PARAMS.ATOM_FEATURES['chiral_tag']) + onek_encoding_unk(
                        int(atom.GetTotalNumHs()),
                        PARAMS.ATOM_FEATURES['num_Hs']) + onek_encoding_unk(
                            int(atom.GetHybridization()),
                            PARAMS.ATOM_FEATURES['hybridization']
                        ) + [1 if atom.GetIsAromatic() else 0] + [
                            atom.GetMass() * 0.01
                        ]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features


def generate_bond_features(bond: Chem.rdchem.Bond,
                           PARAMS: FeaturizationParameters):
    """
    This function generates the feature vector for a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond object.
    PARAMS : deepchem.feat.FeaturizationParameters
        Featurization parameters.

    Returns
    -------
    list
        Feature vector.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles("CC")
    >>> PARAMS = dc.feat.FeaturizationParameters()
    >>> for bond in mol.GetBonds():
    ...     print(dc.feat.generate_bond_features(bond, PARAMS = PARAMS))
    [0, True, False, False, False, False, False, 1, 0, 0, 0, 0, 0, 0]
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def remove_wildcard_atoms(rwmol: Chem.rdchem.RWMol):
    """
    This function removes the connection virtual atoms for open bonds in a molecule.
    This is necessary for molecules with wildcard notations.

    Parameters
    ----------
    rwmol : rdkit.Chem.rdchem.RWMol
        Read and Writable RDKit molecule object.

    Returns
    -------
    rdkit.Chem.rdchem.RWMol
        Read and writable RDKit molecule object.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles("[*]CC")
    >>> rwmol = Chem.RWMol(mol)
    >>> rwmol = dc.utils.remove_wildcard_atoms(rwmol)
    >>> Chem.MolToSmiles(rwmol)
    'CC'
    """
    indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    while len(indices) > 0:
        rwmol.RemoveAtom(indices[0])  # removing the wildcard atom
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)
    return rwmol
