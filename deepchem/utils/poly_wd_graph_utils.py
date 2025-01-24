from rdkit import Chem
import numpy as np
from typing import List, Tuple
import re


def handle_hydrogen(smiles: str,
                    keep_h: bool = False,
                    add_h: bool = False) -> Chem.rdchem.Mol:
    """
    Builds an RDKit molecule from a SMILES string by conditionally
    handling addition or removal of hydrogens.

    This function is useful to process organic molecules properly.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol = dc.utils.handle_hydrogen('C', keep_h=True, add_h=False)
    >>> mol.GetNumAtoms()
    1
    >>> mol = dc.utils.handle_hydrogen('C', keep_h=True, add_h=True)
    >>> mol.GetNumAtoms()
    5

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
    fragment_weights: List[float],
    keep_h: bool,
    add_h: bool,
) -> Chem.rdchem.Mol:
    """
    Builds an RDKit joined molecule from a SMILES string of monomer molecules.
    The weight of each monomer is stored as a metadata property of each atom.

    Examples
    --------
    >>> import deepchem as dc
    >>> mol = dc.utils.make_polymer_mol('C.C', [1, 1], True, False)
    >>> for atom in mol.GetAtoms():
    ...     print(atom.GetDoubleProp('w_frag'))
    1.0
    1.0

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


def parse_polymer_rules(rules: List[str]) -> Tuple[List[tuple], float]:
    """
    This function extracts probabilty weight distribution details for bonds
    from string to list of tuples in following format.
    (start, end, weight_forward, weight_reverse)
    The start and end is in string. (As it will be mapped with open bond string)
    The weight_forward and weight_reverse in float
    It also returns the degree of polymerization of the polymer (DoP) as 1. + np.log10(DoP).
    The DoP can be mentioned at the end of the rules separated by "~".
    If DoP is not given, the function assumes DoP = 1.

    Examples
    --------
    >>> import deepchem as dc
    >>> polymer_info, degree_of_polymerization = dc.utils.parse_polymer_rules(
    ...     ['1-2:0.5:0.5'])
    >>> polymer_info
    [('1', '2', 0.5, 0.5)]
    >>> degree_of_polymerization
    1.0

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
    """

    polymer_info = []

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
        if len(rule.split(":")[0].split("-")) != 2:
            raise ValueError(
                f'incorrect format for bond index mentioning "{rule}"')
        idx1, idx2 = rule.split(':')[0].split('-')
        w12 = float(rule.split(':')[1])  # weight for bond R_idx1 -> R_idx2
        w21 = float(rule.split(':')[2])  # weight for bond R_idx2 -> R_idx1
        polymer_info.append((idx1, idx2, w12, w21))

    return polymer_info, 1. + np.log10(Xn)


def tag_atoms_in_repeating_unit(
        mol: Chem.rdchem.RWMol) -> Tuple[Chem.rdchem.RWMol, dict]:
    """
    This function tags atoms that are part of the core units, as well as atoms
    serving to identify attachment points. In addition, create a map of bond
    types based on what bonds are connected to R groups in the input. The input molecules must be
    of `Chem.rdchem.RWMol` type to be editable.

    Examples
    --------
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> rw_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles('[1*]CC.C[2*]'))
    >>> mol, _ = dc.utils.tag_atoms_in_repeating_unit(rw_mol)
    >>> mol.GetAtomWithIdx(0).GetBoolProp('core')
    False
    >>> mol.GetAtomWithIdx(1).GetBoolProp('core')
    True
    >>> mol.GetAtomWithIdx(3).GetProp('R')
    '2*'

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.RWMol
        RDKit read and write enabled molecule object.

    Returns
    -------
    rdkit.Chem.rdchem.RWMol
        RDKit read and write enabled molecule object.
    dict
        Map of R group to bond type.
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

    Examples
    --------
    >>> import deepchem as dc
    >>> dc.utils.onek_encoding_unk(1, [1, 2, 3])
    [1, 0, 0, 0]
    >>> dc.utils.onek_encoding_unk(69, [1, 2, 3])
    [0, 0, 0, 1]

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
    """

    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def remove_wildcard_atoms(rwmol: Chem.rdchem.RWMol) -> Chem.rdchem.RWMol:
    """
    This function removes the connection virtual atoms for open bonds in a molecule.
    This is necessary for molecules with wildcard notations.

    Examples
    --------
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("[*]CC")
    >>> rwmol = Chem.RWMol(mol)
    >>> rwmol = dc.utils.remove_wildcard_atoms(rwmol)
    >>> Chem.MolToSmiles(rwmol)
    'CC'

    Parameters
    ----------
    rwmol : rdkit.Chem.rdchem.RWMol
        Read and Writable RDKit molecule object.

    Returns
    -------
    rdkit.Chem.rdchem.RWMol
        Read and writable RDKit molecule object.
    """
    indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    while len(indices) > 0:
        rwmol.RemoveAtom(indices[0])  # removing the wildcard atom
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)
    return rwmol


class PolyWDGStringValidator():
    """
    Class for validating the string format of weighted directed graph
    data.This class provides methods to validate the format of a
    datapoint string. This is a specific string format that is used
    for storing weighted directed polymer data in a parsable format.

    The format is as follows:
        [monomer1].[monomer2]|[fraction_of_monomer1]|[fraction_of_monomer2]|<[polymer_rule1]<[polymer_rule2]
    The polymer rule has an own format in it. Which is as follows:
        [[atom_index1]-[atom_index2]]:[fraction_of_bond_between_atom1_to_atom2]:[fraction_of_bond_between_atom2_to_atom1]

    This format is explicitly used for formatting the input for
    Weighted Directed Message Passing Neural Networks (wD-MPNN).
    The input format holds a SMART notation and regular expression
    formatting to keep molecular data with corresponding bonds and
    weights. Irrespective of this explicit usecase, the formatting
    can allow featurization of same data for other graph based neural
    networks.

    The validate method validates the proper formatting for monomer
    molecules, proper value of the fractions and valid atom indicies
    and corresponding weights in the polymer rules.

    Example
    -------
    >>> from deepchem.utils import PolyWDGStringValidator
    >>> validator = PolyWDGStringValidator()
    >>> validator.validate("[1*]C.C[2*]|0.5|0.5|<1-2:0.5:0.5")
    True

    References
    ----------
    .. [1] Aldeghi, Matteo, and Connor W. Coley. "A graph representation of molecular
        ensembles for polymer property prediction." Chemical Science 13.35 (2022): 10486-10498.
    """

    @staticmethod
    def get_parsed_vals(datapoint: str) -> Tuple[str, list, str]:
        """
        This static method parses the datapoint string into 3 parts:
        1. Monomer molecules
        2. Fragments
        3. Polymer rules

        Parameters
        ----------
        datapoint : str
            The datapoint string to parse

        Returns
        -------
        Tuple[str, list, str]
            A tuple containing the 3 parts of the datapoint string
        """
        base_parsed = datapoint.split("|")
        if len(base_parsed) < 3:
            raise ValueError(
                f"Invalid datapoint format: At least 3 splits should be there but found {len(base_parsed)} no. of splits"
            )
        monomer_mols = base_parsed[0]
        polymer_rules = base_parsed[-1]
        fragments = base_parsed[1:-1]
        return monomer_mols, fragments, polymer_rules

    @staticmethod
    def get_polymer_rules(rules_str: str) -> List[str]:
        """
        This static method parses the polymer rules string into a list of rules.

        Parameters
        ----------
        rules_str : str
            The polymer rules string to parse

        Returns
        -------
        List[str]
            A list containing the parsed rule strings
        """
        if len(rules_str.split("<")) == 1:
            raise ValueError(
                "Invalid rules string: The rule string must contain '<' as a separator for rules !"
            )
        return rules_str.split("<")[1:]

    def _validate_fragments(self, datapoint: str):
        """
        This method validate the number of fragments match
        the number of monomers.

        Parameters
        ----------
        datapoint : str
            The datapoint string to validate

        Raises
        ------
        ValueError
            If the number of fragments does not match the number of monomers
        """
        monomer_mols, fragments, _ = self.get_parsed_vals(datapoint)
        if len(fragments) != len(monomer_mols.split(".")):
            raise ValueError(
                f"Number of fragments and number of monomers should match. Mismatch -> No. of Fragments {len(fragments)} , No. of Monomers{len(monomer_mols.split('.'))}"
            )

    def _get_all_wildcards(self, text: str) -> List[str]:
        """
        This method returns all the wildcards present in the given string
        representation by using regular expression to detect digits after
        '*'.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[str]
            A list of all wildcards present in the text
        """
        matches = re.findall(r"\d+(?=\*)", text)
        return matches

    def _validate_wildcards(self, datapoint: str):
        """
        This method validates the presence of wildcards in the polymer
        molecules string and ensures that the sequence of the wildcard
        notation is proper.

        Parameters
        ----------
        datapoint : str
            The datapoint string to validate

        Raises
        ------
        ValueError
            If the wildcards are not present in the sequce the maximum
            wildcard value, ValueError is raised.
        """
        monomer_mols, _, _ = self.get_parsed_vals(datapoint)
        max_wildcard = max(
            [int(x) for x in self._get_all_wildcards(monomer_mols)])
        for wildcard in range(1, max_wildcard + 1):
            if str(wildcard) + "*" not in monomer_mols:
                raise ValueError(
                    f"Invalid wildcard format: The wildcard {wildcard} is not present in the monomer molecules string  as per the sequence of the maximum {max_wildcard}!"
                )

    def _validate_polymer_rules(self, datapoint: str):
        """
        This method validates the format of the polymer rules string
        by checking for the presence of the '-' separator between the
        atom indexes, the correct number of splits in the rule string,
        and the validity of the atom indexes present in the monomer
        SMILES. It also checks if the atom indexes are in the correct
        correct count for a valid bond formation.

        Parameters
        ----------
        datapoint : str
            The datapoint string to validate

        Raises
        ------
        ValueError
            If the polymer rules string is invalid, ValueError is raised
            with appropriate error messages
        """
        monomer_mols, _, polymer_rules = self.get_parsed_vals(datapoint)
        polymer_rule_list = self.get_polymer_rules(polymer_rules)
        for rules in polymer_rule_list:
            splits = rules.split(":")
            if len(splits) != 3:
                raise ValueError(
                    f"Invalid polymer rule format: The rule must contain exactly 3 splits ! but found {len(splits)} splits"
                )
            if "-" not in splits[0]:
                raise ValueError(
                    f"Invalid polymer rule format: The bond string between two wildcard index must be seprated by '-', got invalid data {splits[0]}"
                )
            elif len(splits[0].split("-")) != 2 and any(
                    elem == "" for elem in splits[0].split("-")):
                raise ValueError(
                    f"Invalid polymer rule format: The first split must contain exactly 2 splits to depict connection between atom indexes! but found {len(splits[0].split('-'))} splits"
                )
            else:
                for wild_card_index in splits[0].split("-"):
                    if not wild_card_index.isdigit():
                        raise ValueError(
                            f"Invalid polymer rule format: The first split must contain only digits! but found {wild_card_index}"
                        )
                    if wild_card_index not in monomer_mols:
                        raise ValueError(
                            f"Invalid polymer rule format: The first split must contain only valid wild card indexes! but found {wild_card_index} which is not in {monomer_mols}"
                        )

    def validate(self, datapoint: str):
        """
        This method validates the string format of weighted
        directed graph data. To validate the string format
        it checks for following conditions:

        1. The number of fragments and the number of monomer
           molecules should match.
        2. The wild card indexes should be present in the monomer
           molecules string and should be in the correct sequence.
        3. The polymer rules should be in the correct format.
        4. The atom indexes in the polymer rules should be valid
           and present in the monomer molecules string.

        It raises ValueError if the string format is invalid.

        Parameters
        ----------
        datapoint : str
            The datapoint string to validate

        Returns
        -------
        bool
            True if the string format is valid, None otherwise
            (Error will be raised otherwise)
        """
        self._validate_fragments(datapoint)
        self._validate_wildcards(datapoint)
        self._validate_polymer_rules(datapoint)
        return True
