import pytest
import unittest
from typing import Set

from deepchem.utils.batch_utils import create_input_array

try:
    import torch
    from deepchem.models.torch_models.seqtoseq import SeqToSeq
    has_torch = True
except:
    has_torch = False

# Dataset of SMILES strings for testing SeqToSeq models.
train_smiles = [
    'Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C',
    'Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1',
    'COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O)oc1ccccc12',
    'O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1ccccc1',
    'NC(=O)NC(Cc1ccccc1)C(=O)O', 'CCn1c(CSc2nccn2C)nc2cc(C(=O)O)ccc21',
    'CCc1cccc2c1NC(=O)C21C2C(=O)N(Cc3ccccc3)C(=O)C2C2CCCN21',
    'COc1ccc(C2C(C(=O)NCc3ccccc3)=C(C)N=C3N=CNN32)cc1OC',
    'CCCc1cc(=O)nc(SCC(=O)N(CC(C)C)C2CCS(=O)(=O)C2)[nH]1',
    'CCn1cnc2c1c(=O)n(CC(=O)Nc1cc(C)on1)c(=O)n2Cc1ccccc1'
]

tokens: Set[str] = set()
for s in train_smiles:
    tokens = tokens.union(set(c for c in s))
token_list = sorted(list(tokens))

batch_size = len(train_smiles)

max_length = max(len(s) for s in train_smiles)


class TestSeqToSeq(unittest.TestCase):

    @pytest.mark.torch
    def test_seqtoseq(self):
        """Test the SeqToSeq Class."""
        global token_list
        token_list = token_list + [" "]
        input_dict = dict((x, i) for i, x in enumerate(token_list))
        n_tokens = len(token_list)
        embedding_dimension = 16

        model = SeqToSeq(n_tokens, n_tokens, max_length, batch_size,
                         embedding_dimension)
        inputs = create_input_array(train_smiles, max_length, False, batch_size,
                                    input_dict, " ")
        output, embeddings = model([torch.tensor(inputs), torch.tensor([1])])

        assert output.shape == (batch_size, max_length, n_tokens)
        assert embeddings.shape == (1, batch_size, embedding_dimension)
