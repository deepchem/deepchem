from bert_tokenizer import BertFeaturizer
from transformers import BertTokenizerFast
from icecream import ic

class run_bert():
  def setUp(self):
    self.sequence = '[CLS] D L I P T S S K L V V L D T S L Q V K K A F F A L V T [SEP]'
    self.featurizer = BertFeaturizer.from_pretrained(
      "Rostlab/prot_bert", do_lower_case=False)
    ic(self.featurizer.__repr__)

  def test_call(self):
    """Test BertFeaturizer.__call__(), which is based on BertTokenizerFast."""
    embedding = self.featurizer(self.sequence, return_tensors='pt')
    ic(embedding)
    """
    assert 'input_ids' in embedding and 'attention_mask' in embedding
    assert len(embedding['input_ids']) == 2 and len(embedding['attention_mask']) == 2
    """

  def test_featurize(self):
    """Test that BertFeaturizer.featurize() correctly featurizes all sequences,
    correctly outputs input_ids and attention_mask.
    """
    feats = self.featurizer.featurize(self.sequence)
    """
    assert (len(feats) == 2)
    assert (all([len(f) == 2 for f in feats]))
    """

    long_feat = self.featurizer.featurize(self.sequence)
    """
    assert (len(long_feat) == 1)
    assert (len(long_feat[0] == 2))
    """

if __name__ == "__main__":
  obj = run_bert()
  obj.setUp()
  obj.test_call()
