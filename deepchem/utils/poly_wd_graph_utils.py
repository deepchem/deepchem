from typing import List, Tuple
import re


class PolyWDGStringValidator():
    """
    Class for validating the string format of weighted directed graph data

    Example
    -------
    >>> from deepchem.utils import PolyWDGStringValidator
    >>> validator = PolyWDGStringValidator()
    >>> validator.validate("[1*]C.C[2*]|0.5|0.5|<1-2:0.5:0.5")
    True
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
                f"Invalid datapoint format: At least 3 splits should be there but found {len(base_parsed)} no. of splist"
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
        monomer_mols, fragments, _ = self.get_parsed_vals(datapoint)
        if len(fragments) != len(monomer_mols.split(".")):
            raise ValueError(
                f"Number of fragments and number of monomers should match. Mismatch -> No. of Fragments {len(fragments)} , No. of Monomers{len(monomer_mols.split('.'))}"
            )

    def _get_all_wildcards(self, text: str) -> List[str]:
        matches = re.findall(r"\d+(?=\*)", text)
        return matches

    def _validate_wildcards(self, datapoint: str):
        monomer_mols, _, _ = self.get_parsed_vals(datapoint)
        max_wildcard = max(
            [int(x) for x in self._get_all_wildcards(monomer_mols)])
        for wildcard in range(1, max_wildcard + 1):
            if str(wildcard) + "*" not in monomer_mols:
                raise ValueError(
                    f"Invalid wildcard format: The wildcard {wildcard} is not present in the monomer molecules string  as per the sequence of the maximum {max_wildcard}!"
                )

    def _validate_polymer_rules(self, datapoint: str):
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
        This method validates the string format of weighted directed graph data.
        It raises ValueError if the string format is invalid.

        Parameters
        ----------
        datapoint : str
            The datapoint string to validate
        
        Returns
        -------
        bool
            True if the string format is valid, None otherwise (Error will be raised otherwise)
        """
        self._validate_fragments(datapoint)
        self._validate_wildcards(datapoint)
        self._validate_polymer_rules(datapoint)
        return True
