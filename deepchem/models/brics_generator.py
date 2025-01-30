from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild
from typing import Tuple
from rdkit import Chem


class BRICSGenerator():
    """BRICSGenerator
    BRICS-based molecule generator for organic candidates, polymers and dendrimers.
    This class implements BRICS (Breaking Retrosynthetically Interesting Chemical Substructures)
    fragmentation and recombination to generate new molecular structures, with special support
    for polymers and dendrimers [1]_. The generator is explicitly used for polymer generation
    purposes mentioned in the research paper "Open-source Polymer Generative Pipeline" [2]_.
    References
    ----------
    [1] Liu, Tairan, et al. "Break down in order to build up: decomposing small
        molecules for fragment-based drug design with e molfrag." Journal of chemical
        information and modeling 57.4 (2017): 627-631.
    [2] Mohanty, Debasish, et al. "Open-source Polymer Generative Pipeline."
        arXiv preprint arXiv:2412.08658 (2024).
    Examples
    --------
    >>> generator = BRICSGenerator(verbose=True)
    # Generate new molecules from SMILES
    >>> smiles = ['CC(=O)Oc1ccccc1C(=O)O']
    >>> new_mols, count = generator.sample(smiles)
    # Generate polymers from polymer SMILES
    >>> psmiles = ['*CC(=O)CC*']
    >>> polymers, count = generator.sample(psmiles, is_polymer=True)
    """

    def __init__(self, verbose: bool = False) -> None:
        self.input_types = ["smiles", "psmiles"]
        self.verbose = verbose

    def _BRICS_decompose(self, smiles_list: list) -> list:
        """Decompose input molecules using BRICS fragmentation rules.

        Parameters
        ----------
        smiles_list : list
            List of SMILES strings to decompose

        Returns
        -------
        list
            List of SMILES strings representing molecular fragments

        Examples
        --------
        >>> generator = BRICSGenerator()
        >>> fragments = generator._BRICS_decompose(['CC(=O)Oc1ccccc1C(=O)O'])
        """
        break_repo = []
        if self.verbose:
            print("[+] Incoming molecules ...")
            print("[+] Decomposing the molecules ...")
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            dec = list(BRICSDecompose(mol))
            break_repo.extend(dec)
        if self.verbose:
            print("[+] Decomposed the molecules ...")
        return break_repo

    def _BRICS_build(self, decomposed_list: list) -> list:
        """Recombine molecular fragments using BRICS rules.

        Parameters
        ----------
        decomposed_list : list
            List of SMILES strings representing fragments

        Returns
        -------
        list
            List of SMILES strings for newly generated molecules

        Examples
        --------
        >>> generator = BRICSGenerator()
        >>> fragments = ['CCO', 'C(=O)O']
        >>> new_mols = generator._BRICS_build(fragments)
        """
        mol_list = [Chem.MolFromSmiles(dec) for dec in decomposed_list]
        if self.verbose:
            print("[+] Building the molecules ...")
        build = list(BRICSBuild(mol_list))
        if self.verbose:
            print("[+] Built the molecules ...")
        return [Chem.MolToSmiles(mol) for mol in build]

    def replace_wildcards_with_vatoms(self, psmiles_list: list) -> list:
        """Replace wildcard atoms (*) with virtual atoms [At] for polymer processing.

        Parameters
        ----------
        psmiles_list : list
            List of polymer SMILES containing wildcards
        Returns
        -------
        list
            Modified SMILES with [At] replacing *
        Raises
        ------
        ValueError
            If any SMILES string has fewer than 2 wildcard notation "[*]
        Examples
        --------
        >>> generator = BRICSGenerator()
        >>> modified = generator.replace_wildcards_with_vatoms(['*CC*'])
        >>> print(modified)  # ['[At]CC[At]']
        """
        mod_list = []
        for psmiles in psmiles_list:
            if psmiles.count("*") < 2:
                raise ValueError(
                    f"The number of wildcards should be atleast 2, which is not the case for input {psmiles}"
                )
            if psmiles.count("*") == psmiles.count("[*]"):
                mod_list.append(psmiles.replace("[*]", "[At]"))
            else:
                mod_list.append(psmiles.replace("*", "[At]"))
        return mod_list

    def replace_vatoms_with_wildcards(self, psmiles_list: list) -> list:
        """Replace virtual atoms ([At]) with wildcards (*) in polymer SMILES.
        This method converts internal virtual atom representation back to standard
        polymer SMILES notation with wildcards.
        Parameters
        ----------
        psmiles_list : list
            List of SMILES strings containing [At] virtual atoms
        Returns
        -------
        list
            Modified SMILES with [At] replaced by [*]

        Examples
        --------
        >>> generator = BRICSGenerator()
        >>> modified = generator.replace_vatoms_with_wildcards(['[At]CC[At]'])
        >>> print(modified)  # ['[*]CC[*]']
        >>> # Multiple virtual atoms
        >>> result = generator.replace_vatoms_with_wildcards(['[At]CC([At])CC[At]'])
        >>> print(result)  # ['[*]CC([*])CC[*]']

        Raises
        ------
        ValueError
            If any SMILES string has fewer than 2 virtual atoms
        """
        mod_list = []
        for psmiles in psmiles_list:
            if psmiles.count("[At]") < 2:
                raise ValueError(
                    f"The number of vritual atoms should be atleast 2, which is not the case for input {psmiles}"
                )
            mod_list.append(psmiles.replace("[At]", "[*]"))
        return mod_list

    def filter_candidates(self,
                          gen_mol_list: list,
                          is_polymer: bool = False,
                          is_dendrimer: bool = False) -> list:
        """Filter generated molecules based on polymer/dendrimer criteria.

        Parameters
        ----------
        gen_mol_list : list
            List of generated SMILES strings
        is_polymer : bool, optional
            Filter for polymer structures (2 connection points)
        is_dendrimer : bool, optional
            Filter for dendrimer structures (3+ connection points)

        Returns
        -------
        list
            Filtered list of SMILES strings

        Raises
        ------
        ValueError
            The dendrimer selection should not be enabled with
            polymer selecting being disabled

        Examples
        --------
        >>> generator = BRICSGenerator()
        >>> mols = ['[At]CC[At]', '[At]CC[At]CC[At]']
        >>> polymers = generator.filter_candidates(mols, is_polymer=True)
        """
        if is_polymer and is_dendrimer:
            filtered_mols = list(
                filter(lambda mol: mol.count("[At]") >= 2, gen_mol_list))
        elif is_polymer and not is_dendrimer:
            filtered_mols = list(
                filter(lambda mol: mol.count("[At]") == 2, gen_mol_list))
        elif not is_polymer and is_dendrimer:
            raise ValueError(
                "The dendrimer selection is available along with polymer selection only !"
            )
        if self.verbose:
            print("[+] Filtering the molecules ...")
        return filtered_mols

    def sample(self,
               smiles_list: list,
               is_polymer: bool = False,
               is_dendrimer: bool = False) -> Tuple[list, int]:
        """Generate new molecules through BRICS decomposition and recombination.

        Parameters
        ----------
        smiles_list : list
            Input SMILES/pSMILES strings
        is_polymer : bool, optional
            Process as polymer SMILES
        is_dendrimer : bool, optional
            Generate dendrimer structures
        Returns
        -------
        Tuple[list, int]
            Generated SMILES strings and count
        Examples
        --------
        >>> generator = BRICSGenerator()
        # Small molecule generation
        >>> new_mols, count = generator.sample(['CC(=O)O'])
        # Polymer generation
        >>> polymers, count = generator.sample(['*CC*'], is_polymer=True)
        # Dendrimer generation
        >>> dendrimers, count = generator.sample(['*CC(*)CC*'],
        ...                                      is_polymer=True,
        ...                                      is_dendrimer=True)
        """
        if is_polymer:
            smiles_list = self.replace_wildcards_with_vatoms(smiles_list)
        decomposed_list = self._BRICS_decompose(smiles_list)
        composed_candidates = self._BRICS_build(decomposed_list)
        if is_polymer or is_dendrimer:
            filtered_mols = self.filter_candidates(composed_candidates,
                                                   is_polymer, is_dendrimer)
        else:
            filtered_mols = composed_candidates
        if self.verbose:
            print(
                f"[+] total {len(filtered_mols)} are created in the process !")
        return filtered_mols, len(filtered_mols)
