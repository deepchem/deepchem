import unittest
from deepchem.feat import BertFeaturizer
from transformers import BertTokenizerFast


class TestBertFeaturizer(unittest.TestCase):
  """Tests for BertFeaturizer, based on tests for RobertaFeaturizer and
  usage examples in Rostlab prot_bert documentation hosted by HuggingFace."""

  def setUp(self):
    self.sequence = ['[CLS] D L I P T S S K L V [SEP]', '[CLS] V K K A F F A L V T [SEP]']
    self.sequence_long = ['[CLS] D L I P T S S K L V V K K A F F A L V T [SEP]']
    self.featurizer = BertFeaturizer.from_pretrained(
        "Rostlab/prot_bert", do_lower_case=False)

  def test_call(self):
    """Test BertFeaturizer.__call__(), which is based on BertTokenizerFast."""
    embedding = self.featurizer(
        self.sequence, return_tensors='pt')
    embedding_long = self.featurizer(
      self.sequence_long * 2, return_tensors='pt')
    for emb in [embedding, embedding_long]:
      assert 'input_ids' in emb.keys() and 'attention_mask' in emb.keys()
      assert len(embedding['input_ids']) == 2 and len(emb['attention_mask']) == 2

  def test_featurize(self):
    """Test that BertFeaturizer.featurize() correctly featurizes all sequences,
    correctly outputs input_ids and attention_mask.
    """
    feats = self.featurizer.featurize(self.sequence)
    assert (len(feats) == 2)
    assert (all([len(f) == 2 for f in feats]))

    long_feat = self.featurizer.featurize(
        self.sequence)
    assert (len(long_feat) == 1)
    assert (len(long_feat[0] == 2))
