import pytest
from transformers import RobertaTokenizerFast


@pytest.mark.torch
def test_smiles_call():
  """Test __call__ method for the featurizer, which is inherited from HuggingFace's RobertaTokenizerFast"""
  from deepchem.feat.roberta_tokenizer import RobertaFeaturizer
  smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
  long_molecule_smiles = [
      "CCCCCCCCCCCCCCCCCCCC(=O)OCCCNC(=O)c1ccccc1SSc1ccccc1C(=O)NCCCOC(=O)CCCCCCCCCCCCCCCCCCC"
  ]
  tokenizer = RobertaTokenizerFast.from_pretrained(
      "seyonec/SMILES_tokenized_PubChem_shard00_160k", do_lower_case=False)
  featurizer = RobertaFeaturizer(tokenizer)
  embedding = featurizer(smiles, add_special_tokens=True, truncation=True)
  embedding_long = featurizer(
      long_molecule_smiles * 2, add_special_tokens=True, truncation=True)
  for emb in [embedding, embedding_long]:
    assert 'input_ids' in emb.keys() and 'attention_mask' in emb.keys()
    assert len(emb['input_ids']) == 2 and len(emb['attention_mask']) == 2


@pytest.mark.torch
def test_smiles_featurize():
  """Test the .featurize method, which will convert the dictionary output to an array

    Checks that all SMILES are featurized and that each featurization
    contains input_ids and attention_mask
    """
  from deepchem.feat.roberta_tokenizer import RobertaFeaturizer
  smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)N1CN(C(C)=O)C(O)C1O"]
  long_molecule_smiles = [
      "CCCCCCCCCCCCCCCCCCCC(=O)OCCCNC(=O)c1ccccc1SSc1ccccc1C(=O)NCCCOC(=O)CCCCCCCCCCCCCCCCCCC"
  ]
  tokenizer = RobertaTokenizerFast.from_pretrained(
      "seyonec/SMILES_tokenized_PubChem_shard00_160k", do_lower_case=False)
  featurizer = RobertaFeaturizer(tokenizer)
  feats = featurizer.featurize(smiles, add_special_tokens=True, truncation=True)
  assert (len(feats) == 2)
  assert (all([len(f) == 2 for f in feats]))
  long_feat = featurizer.featurize(
      long_molecule_smiles, add_special_tokens=True, truncation=True)
  assert (len(long_feat) == 1)
  assert (len(long_feat[0] == 2))
