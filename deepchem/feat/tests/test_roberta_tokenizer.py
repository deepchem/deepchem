from deepchem.feat import RobertaFeaturizer


class TestRobertaFeaturizer(unittest.TestCase):
  """Tests for RobertaFeaturizer"""

  def setUp(self):
    self.smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
    self.long_molecule_smiles = [
        "CCCCCCCCCCCCCCCCCCCC(=O)OCCCNC(=O)c1ccccc1SSc1ccccc1C(=O)NCCCOC(=O)CCCCCCCCCCCCCCCCCCC"
    ]
    self.tokenizer = RobertaFeaturizer.from_pretrained(
        "seyonec/SMILES_tokenized_PubChem_shard00_160k")

  def test_smiles_call(self):
    embedding = self.tokenizer(
        self.smiles, add_special_tokens=True, truncation=True)
    embedding_long = self.tokenizer(
        self.long_molecule_smiles * 2, add_special_tokens=True, truncation=True)
    for emb in [embedding, embedding_long]:
      assert 'input_ids' in emb.keys and 'attention_mask' in emb.keys
      assert len(emb['input_ids'] == 2) and len(emb['attention_mask'] == 2)

  def test_smiles_featurize(self):
    feats = self.tokenizer(
        self.smiles, add_special_tokens=True, truncation=True)
    assert (len(feats) == 2)
    assert (all([len(f) == 2 for f in feats]))
