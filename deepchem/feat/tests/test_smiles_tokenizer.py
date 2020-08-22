# Requriments - transformers, tokenizers

from unittest import TestCase
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from transformers import RobertaForMaskedLM


class TestSmilesTokenizer(TestCase):
  """Tests the SmilesTokenizer to load the USPTO vocab file and a ChemBERTa Masked LM model with pre-trained weights.."""


  def test_tokenize(self):
      model = RobertaForMaskedLM.from_pretrained('seyonec/SMILES_tokenized_PubChem_shard00_50k')
      model.num_parameters()

      tokenizer = SmilesTokenizer('deepchem/feat/tests/data/vocab.txt', max_len=model.config.max_position_embeddings)
      print(tokenizer.encode("CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@](C#N)([C@H](O)[C@@H]1O)C1=CC=C2N1N=CN=C2N)OC1=CC=CC=C1"))

