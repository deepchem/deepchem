import pytest


@pytest.mark.torch
def test_featurize():
  """Test that RxnFeaturizer.featurize() correctly featurizes the reactions,
    correctly outputs the input_ids and attention_mask.
    """
  from deepchem.feat.reaction_featurizer import RxnFeaturizer
  from transformers import RobertaTokenizerFast
  tokenizer = RobertaTokenizerFast.from_pretrained(
      "seyonec/PubChem10M_SMILES_BPE_450k")
  featurizer = RxnFeaturizer(tokenizer, sep_reagent=True)
  reaction = ['CCS(=O)(=O)Cl.OCCBr>CCN(CC)CC.CCOCC>CCS(=O)(=O)OCCBr']
  feats = featurizer.featurize(reaction)
  assert (feats.shape == (1, 2, 2, 1))


@pytest.mark.torch
def test_separation():
  """Tests the reagent separation feature after tokenizing the reactions.
    The tokenized reaction is decoded before testing for equality, to make the
    test more readable.
    """
  from deepchem.feat.reaction_featurizer import RxnFeaturizer
  from transformers import RobertaTokenizerFast
  tokenizer = RobertaTokenizerFast.from_pretrained(
      "seyonec/PubChem10M_SMILES_BPE_450k")
  featurizer_mix = RxnFeaturizer(tokenizer, sep_reagent=False)
  featurizer_sep = RxnFeaturizer(tokenizer, sep_reagent=True)
  reaction = ['CCS(=O)(=O)Cl.OCCBr>CCN(CC)CC.CCOCC>CCS(=O)(=O)OCCBr']
  feats_mix = featurizer_mix.featurize(reaction)
  feats_sep = featurizer_sep.featurize(reaction)

  # decode the source in the mixed and separated cases
  mix_decoded = tokenizer.decode(feats_mix[0][0][0][0])
  sep_decoded = tokenizer.decode(feats_sep[0][0][0][0])
  assert mix_decoded == '<s>CCS(=O)(=O)Cl.OCCBr.CCN(CC)CC.CCOCC></s>'
  assert sep_decoded == '<s>CCS(=O)(=O)Cl.OCCBr>CCN(CC)CC.CCOCC</s>'
