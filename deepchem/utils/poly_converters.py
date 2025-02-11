from typing import Optional, Tuple
from rdkit import Chem
import re
from rdkit.Chem import rdmolops

ALLOWED_CONVERSION_TYPES = ["alternate", "block", "random"]


class PSMILES2WDGConverter:
    """
    This class is used to convert a PSMILES string to a to corresponding WDGraph string.
    The conversion can be done in two ways:
        1. With metadata: The metadata contains the indices of the atoms to be bonded and the residue.
        2. Without metadata: The conversion is done using the conversion types specified in the constructor.
    The mechanism is as follows:
        1. The PSMILES string is split into two parts.
        2. The bond is broken between the specified indices.
        3. The fragments are converted to SMILES strings.
        4. The SMILES strings are combined with the residue to form the final WDGraph string.
    This specific mechanism is utilized to convert a PSMILES string to a WDGraph string such that the representational
    variation is tested with neural network architectural differences in our research paper "Open-Source Polymer Generative Pipeline" [1]_.

    References
    ----------
    .. [1] Mohanty, Debasish, et al. "Open-source Polymer Generative Pipeline."
        arXiv preprint arXiv:2412.08658 (2024).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from deepchem.utils.poly_converters import PSMILES2WDGConverter
    >>> psmiles = "*CCCC*"
    >>> metadata = {
    ...     "indicies": [2, 3],
    ...     "seq_index": 4,
    ...     "residue": "0.5|0.5|",
    ...     "smiles_type": "SMARTS"
    ... }
    >>> converter = PSMILES2WDGConverter()
    >>> result = converter([psmiles], [metadata])
    >>> print(result)
    [['[*:1]CC.[*:2]C[*:3][*:4]C|0.5|0.5|']]
    """

    def __init__(self, conversion_types: list = ["alternate"]) -> None:
        """
        Parameters
        ----------
        conversion_types: list
            The conversion types to be used for the conversion. The default is ["alternate"].
        """
        for conversion in conversion_types:
            if conversion not in ALLOWED_CONVERSION_TYPES:
                raise ValueError(
                    f"The conversion type {conversion} is not supported choose any from {ALLOWED_CONVERSION_TYPES}"
                )
        self.conversion_types = conversion_types
        self.CONVERT_OUTRO_MAPPER = {
            "alternate":
                "<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5",
            "block":
                "<1-2:0.75:0.75<3-4:0.75:0.75<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125",
            "random":
                "<1-2:0.50:0.50<3-4:0.50:0.50<1-3:0.33:0.33<1-4:0.33:0.33<2-3:0.33:0.33<2-4:0.33:0.33"
        }

    def index_wildcards(self, psmiles_part: str) -> str:
        """
        This function is used to index the wildcard atoms in the PSMILES string.

        Parameters
        ----------
        psmiles_part: str
            The PSMILES string to be indexed.

        Returns
        -------
        str
            The indexed PSMILES string.
        """
        counter = 1
        mod_psmiles = ""
        for idx, char in enumerate(psmiles_part):
            if char == "*":
                mod_psmiles += "[" + str(counter) + "*" + "]"
                counter += 1
            else:
                mod_psmiles += char
        return mod_psmiles

    def add_indicies_to_smiles_from_meta(self, smiles: str,
                                         seq_index: int) -> str:
        """
        This function is used to add the indices to the SMILES string from the metadata.

        Parameters
        ----------
        smiles: str
            The SMILES string to be modified.
        seq_index: int
            The index of the sequence in the SMILES string.

        Returns
        -------
        str
            The modified SMILES string.
        """
        restored_smiles = smiles[:seq_index] + "**" + smiles[seq_index:]
        return restored_smiles

    def make_wdgraph_string_from_meta(self, psmiles: str,
                                      indicies: list) -> str:
        """
        This function is used to make the WDGraph string from the metadata.

        Parameters
        ----------
        psmiles: str
            The PSMILES string to be converted.
        indicies: list
            The indices of the atoms to be bonded.

        Returns
        -------
        str
            The WDGraph string.
        """
        # Convert the combined SMILES to an RWMol object for editing
        combined_mol = Chem.MolFromSmiles(psmiles)
        editable_mol = Chem.RWMol(combined_mol)

        # Break the bond between the specified indices
        editable_mol.RemoveBond(indicies[0], indicies[1])

        # Generate fragments (separate molecules) after breaking the bond
        fragments = Chem.GetMolFrags(editable_mol, asMols=True)

        # Convert each fragment to its respective SMILES string
        fragment_smiles = [Chem.MolToSmiles(frag) for frag in fragments]
        return ".".join(fragment_smiles)

    def convert_smiles_to_SMARTS(self, smiles_string: str) -> str:
        """
        This function is used to convert the SMILES string to SMARTS string.

        Parameters
        ----------
        smiles_string: str
            The SMILES string to be converted.

        Returns
        -------
        str
            The converted SMARTS string.
        """
        counter = 1
        mod_string = ""
        for s in smiles_string:
            if s == "*":
                mod_string += f"[*:{counter}]"
                counter += 1
            else:
                mod_string += s
        return mod_string

    def convert_smiles_to_MOD_SMARTS(self, smiles_string: str) -> str:
        """
        This function is used to convert the SMILES string to MOD SMARTS string.

        Parameters
        ----------
        smiles_string: str
            The SMILES string to be converted.

        Returns
        -------
        str
            The converted MOD SMARTS string.
        """
        counter = 1
        mod_string = ""
        for s in smiles_string:
            if s == "*":
                mod_string += f"[{counter}*]"
                counter += 1
            else:
                mod_string += s
        return mod_string

    def compose_from_meta(self, psmiles: str, metadata: dict) -> str:
        """
        This function is used to compose the WDGraph string from the metadata.

        Parameters
        ----------
        psmiles: str
            The PSMILES string to be converted.
        metadata: dict
            The metadata containing the indices of the atoms to be bonded, the residue, and the type of SMILES string.

        Returns
        -------
        str
            The composed WDGraph string.
        """
        w_idx_psmiles = self.add_indicies_to_smiles_from_meta(
            psmiles, metadata["seq_index"])
        smile_part = self.make_wdgraph_string_from_meta(w_idx_psmiles,
                                                        metadata["indicies"])
        if metadata["smiles_type"] == "SMARTS":
            mod_smiles = self.convert_smiles_to_SMARTS(smile_part)
        elif metadata["smiles_type"] == "MOD_SMARTS":
            mod_smiles = self.convert_smiles_to_MOD_SMARTS(smile_part)
        return mod_smiles + "|" + metadata["residue"]

    def convert(self,
                psmiles_string: str,
                metadata: Optional[dict] = None) -> list:
        """
        This function is used to convert the PSMILES string to a WDGraph string.

        Parameters
        ----------
        psmiles_string: str
            The PSMILES string to be converted.
        metadata: dict
            The metadata containing the indices of the atoms to be bonded, the residue, and the type of SMILES string.

        Returns
        -------
        list
            The converted WDGraph string.
        """
        converted = []
        if metadata:
            converted.append(self.compose_from_meta(psmiles_string, metadata))
        else:
            intro = self.index_wildcards(psmiles_string + "." + psmiles_string)
            midtro = "|0.5|0.5|"
            for conversion_type in self.conversion_types:
                outro = self.CONVERT_OUTRO_MAPPER[conversion_type]
                converted.append(intro + midtro + outro)
        return converted

    def __call__(self,
                 psmiles_list: list,
                 metadata_list: Optional[list] = None) -> list:
        """
        This function is used to call the conversion process.

        Parameters
        ----------
        psmiles_list: list
            The list of PSMILES strings to be converted.
        metadata_list: list
            The list of metadata containing the indices of the atoms to be bonded, the residue, and the type of SMILES string.

        Returns
        -------
        list
            The converted WDGraph strings.
        """
        converted_list = []
        if metadata_list is None:
            for psmiles_string in psmiles_list:
                converted_list.append(self.convert(psmiles_string))
        else:
            for psmiles_string, metadata in zip(psmiles_list, metadata_list):
                converted_list.append(self.convert(psmiles_string, metadata))
        return converted_list


class WDG2PSMILESConverter:
    """
    This class is used to convert a WDGraph string to a corresponding PSMILES string.
    The mechanism returns two formats:
        1. With metadata: Gives metadata containing the indices of the atoms to be bonded and the residue.
        2. Without metadata: Gives only the converted PSMILES string.
           (the exact reverse conversion of PSMILES to WDG will be done considering alternating structure only)
    The mechanism is as follows:
        1. The WDGraph string is split into two parts (PSMILES part, Residue part)
        2. The PSMILES part is converted from SMART notation to normal notation.
        3. The PSMILES part is converted to a single molecule keeping the bond breaking indicies as metadata.
        4. The final PSMILES is returned with the metadata of the bond breaking indices and initial residue.
    This specific mechanism is utilized to convert a PSMILES string to a WDGraph string such that the representational
    variation is tested with neural network architectural differences in our research paper "Open-Source Polymer Generative Pipeline" [1]_.

    References
    ----------
    .. [1] Mohanty, Debasish, et al. "Open-source Polymer Generative Pipeline."
        arXiv preprint arXiv:2412.08658 (2024).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from deepchem.utils.poly_converters import WDG2PSMILESConverter
    >>> wd_graph_string = "[*:1]CC[*:2].[*:3]CC[*:4]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"
    >>> converter = WDG2PSMILESConverter(return_metadata = True)
    >>> result, meta = converter([wd_graph_string])
    >>> print(result)
    ['*CCCC*']
    >>> print(meta)
    [{'indicies': [3, 4], 'seq_index': 3, 'residue': '0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5', 'smiles_type': 'SMARTS'}]
    """

    def __init__(self, return_metadata: bool = True) -> None:
        """
        Parameters
        ----------
        return_metadata: bool
            Whether to return metadata or not. The default is True.
        """
        self.return_metadata = return_metadata

    def get_wildcard_bond_indecies(self,
                                   combined_psmiles: str) -> Tuple[int, int]:
        """
        This function is used to get the wildcard bond indices in the combined PSMILES string.

        Parameters
        ----------
        combined_psmiles: str
            The combined PSMILES string.

        Returns
        -------
        Tuple(int, int)
            The wildcard bond indices.
        """
        if "**" not in combined_psmiles:
            raise ValueError(
                "The combined PSMILES must have two wildcard atoms together !")
        atoms = [
            atom.GetSymbol()
            for atom in Chem.MolFromSmiles(combined_psmiles).GetAtoms()
        ]
        found = False
        for i, atom in enumerate(atoms):
            if atom == "*" and found:
                return i, i + 1
            elif atom == "*":
                found = True
        return -1, -1

    def convert_smiles_part(self, smiles: str) -> Tuple[str, int, int, int]:
        """
        This function is used to convert the SMILES part of the WDG string to a PSMILES string and returns the indicies along side.

        Parameters
        ----------
        smiles: str
            The SMILES part of the WDG string.

        Returns
        -------
        Tuple(str, int, int, int)
            The converted PSMILES string, the bond breaking indicies, and the sequence index.
        """
        # Split the SMILES string into individual molecules
        if '.' not in smiles:
            raise ValueError(
                "The input SMILES string does not contain a joining notation")

        mols = [Chem.MolFromSmiles(s) for s in smiles.split('.')]

        # Make sure both molecules are valid
        if None in mols:
            raise ValueError("One of the SMILES strings is invalid")

        # Combine the two molecules into a single editable molecule
        combined_mol = Chem.CombineMols(mols[0], mols[1])
        combined_mol = Chem.RWMol(combined_mol)

        # Get atom indices of wildcard '*' atoms in the combined molecule
        wildcard_indices = [
            atom.GetIdx()
            for atom in combined_mol.GetAtoms()
            if atom.GetSymbol() == '*'
        ]

        # Check if there are at least two wildcards to form a bond
        if len(wildcard_indices) < 2:
            raise ValueError("Not enough wildcards to form a bond")

        # Select the indices of the wildcards you want to bond
        # For example, let's bond the second and third wildcards (index 1 and 2)
        # Note: Adjust indices as per your needs
        index1 = wildcard_indices[1]
        index2 = wildcard_indices[2]

        # Add a single bond between the selected wildcard atoms
        combined_mol.AddBond(index1, index2, Chem.BondType.SINGLE)

        # Remove the wildcard atoms after bonding
        rdmolops.DeleteSubstructs(combined_mol, Chem.MolFromSmiles('*'))

        # Convert the RWMol back to a Mol object
        final_mol = combined_mol.GetMol()

        # Generate the SMILES string of the final combined molecule
        final_smiles = Chem.MolToSmiles(final_mol)
        index_1, index_2 = self.get_wildcard_bond_indecies(final_smiles)
        seq_index = final_smiles.find("**")
        return final_smiles.replace("**", ""), index_1, index_2, seq_index

    def replace_SMARTS(self, smiles: str) -> str:
        """
        This function is used to replace the SMARTS notation in the SMILES string.

        Parameters
        ----------
        smiles: str
            The SMILES string to be modified.

        Returns
        -------
        str
            The modified SMILES string
        """
        pattern = r"\[\*:\d+\]"
        result = re.sub(pattern, "*", smiles)
        return result

    def replace_mod_SMARTS(self, mod_smiles: str) -> str:
        """
        This function is used to replace the MOD SMARTS notation in the SMILES string.

        Parameters
        ----------
        mod_smiles: str
            The SMILES string to be modified.

        Returns
        -------
        str
            The modified SMILES string.
        """
        pattern = r"\[\d+\*\]"
        result = re.sub(pattern, "*", mod_smiles)
        return result

    def convert(self, wdg_graph_string: str) -> Tuple[str, dict]:
        """
        This is used to convert a single WDGraph string to a PSMILES string and form the metadata.

        Parameters
        ----------
        wdg_graph_string: str
            The WDGraph string to be converted.

        Returns
        -------
        Tuple(str, dict)
            The converted PSMILES string and the metadata.
        """
        parts = wdg_graph_string.split("|")
        monomer_smiles = parts[0]
        smiles_type = None
        residue = "|".join(parts[1:])
        if "*:" in monomer_smiles:
            mod_smiles = self.replace_SMARTS(monomer_smiles)
            smiles_type = "SMARTS"
        elif "*]" in monomer_smiles:
            mod_smiles = self.replace_mod_SMARTS(monomer_smiles)
            smiles_type = "MOD_SMARTS"
        else:
            raise ValueError(
                f"The data {wdg_graph_string} is not valid for this conversion !"
            )
        psmiles, index_1, index_2, seq_index = self.convert_smiles_part(
            mod_smiles)
        if self.return_metadata:
            return psmiles, {
                "indicies": [index_1, index_2],
                "seq_index": seq_index,
                "residue": residue,
                "smiles_type": smiles_type
            }
        else:
            return psmiles, {}

    def __call__(self, wd_graph_list: list) -> Tuple[list, list]:
        """
        This function is used to call the conversion process.

        Parameters
        ----------
        wd_graph_list: list
            The list of WDGraph strings to be converted.

        Returns
        -------
        Tuple(list, list)
            The converted PSMILES strings and the metadata.
        """
        converted_psmiles_list = []
        metadata_list = []
        for wd_graph_string in wd_graph_list:
            if self.return_metadata:
                psmiles, metadata = self.convert(wd_graph_string)
                converted_psmiles_list.append(psmiles)
                metadata_list.append(metadata)
            else:
                psmiles, _ = self.convert(wd_graph_string)
                converted_psmiles_list.append(psmiles)
        if self.return_metadata:
            return converted_psmiles_list, metadata_list
        return converted_psmiles_list, []
