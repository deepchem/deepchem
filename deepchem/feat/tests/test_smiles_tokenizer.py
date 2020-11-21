# Requirements - transformers, tokenizers
import os
from unittest import TestCase
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from transformers import RobertaForMaskedLM


class TestSmilesTokenizer(TestCase):
  """Tests the SmilesTokenizer to load the USPTO vocab file and a ChemBERTa Masked LM model with pre-trained weights.."""

  def test_tokenize(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'data', 'vocab.txt')
    tokenized_smiles = [
        12, 16, 16, 16, 17, 16, 16, 18, 16, 19, 16, 17, 22, 19, 18, 33, 17, 16,
        18, 23, 181, 17, 22, 19, 18, 17, 19, 16, 33, 20, 19, 55, 17, 16, 23, 18,
        17, 33, 17, 19, 18, 35, 20, 19, 18, 16, 20, 22, 16, 16, 22, 16, 21, 23,
        20, 23, 22, 16, 23, 22, 16, 21, 23, 18, 19, 16, 20, 22, 16, 16, 22, 16,
        16, 22, 16, 20, 13
    ]

    model = RobertaForMaskedLM.from_pretrained(
        'seyonec/SMILES_tokenized_PubChem_shard00_50k')
    model.num_parameters()

    tokenizer = SmilesTokenizer(
        vocab_path, max_len=model.config.max_position_embeddings)

    assert tokenized_smiles == tokenizer.encode(
        "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@](C#N)([C@H](O)[C@@H]1O)C1=CC=C2N1N=CN=C2N)OC1=CC=CC=C1"
    )
