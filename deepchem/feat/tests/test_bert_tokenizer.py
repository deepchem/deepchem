import pytest


@pytest.mark.torch
def test_call():
  """Test BertFeaturizer.__call__(), which is based on BertTokenizerFast."""
  from deepchem.feat.bert_tokenizer import BertFeaturizer
  from transformers import BertTokenizerFast
  sequence = [
      '[CLS] D L I P T S S K L V [SEP]', '[CLS] V K K A F F A L V T [SEP]'
  ]
  sequence_long = ['[CLS] D L I P T S S K L V V K K A F F A L V T [SEP]']
  tokenizer = BertTokenizerFast.from_pretrained(
      "Rostlab/prot_bert", do_lower_case=False)
  featurizer = BertFeaturizer(tokenizer)
  embedding = featurizer(sequence, return_tensors='pt')
  embedding_long = featurizer(
      sequence_long * 2, return_tensors='pt')
  for emb in [embedding, embedding_long]:
    assert 'input_ids' in emb.keys() and 'attention_mask' in emb.keys()
    assert len(embedding['input_ids']) == 2 and len(
        emb['attention_mask']) == 2

def test_featurize():
  """Test that BertFeaturizer.featurize() correctly featurizes all sequences,
  correctly outputs input_ids and attention_mask."""
  from deepchem.feat.bert_tokenizer import BertFeaturizer
  from transformers import BertTokenizerFast
  sequence = [
      '[CLS] D L I P T S S K L V [SEP]', '[CLS] V K K A F F A L V T [SEP]'
  ]
  sequence_long = ['[CLS] D L I P T S S K L V V K K A F F A L V T [SEP]']
  tokenizer = BertTokenizerFast.from_pretrained(
      "Rostlab/prot_bert", do_lower_case=False)
  featurizer = BertFeaturizer(tokenizer)
  feats = featurizer.featurize(sequence)
  long_feat = featurizer.featurize(sequence_long)
  assert (len(feats) == 2)
  assert (all([len(f) == 3 for f in feats]))
  assert (len(long_feat) == 1)
  assert (len(long_feat[0] == 2))
