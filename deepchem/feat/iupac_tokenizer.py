import re
from deepchem.feat.base_classes import Featurizer


class IUPACTokenizer(Featurizer):
    """Word-level tokenizer for IUPAC chemical names."""

    def _featurize(self, iupac_name):
        tokens = re.findall(r"[A-Za-z]+|\d+|[-(),]", iupac_name)
        return tokens