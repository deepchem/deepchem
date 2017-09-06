import deepchem as dc
import numpy as np
import unittest


class TestSeqToSeq(unittest.TestCase):

  def test_int_sequence(self):
    """Test learning to reproduce short sequences of integers."""

    sequence_length = 10
    tokens = list(range(10))
    s = dc.models.SeqToSeq(
        tokens,
        tokens,
        sequence_length,
        encoder_layers=2,
        decoder_layers=2,
        embedding_dimension=150,
        learning_rate=0.01,
        dropout=0.1)

    # Train the model on random sequences.  We aren't training long enough to
    # really make it reliable, but I want to keep this test fast, and it should
    # still be able to reproduce a reasonable fraction of input sequences.

    def generate_sequences(num_sequences):
      for i in range(num_sequences):
        seq = [
            np.random.randint(10)
            for x in range(np.random.randint(1, sequence_length + 1))
        ]
        yield (seq, seq)

    s.fit_sequences(generate_sequences(25000))

    # Test it out.

    count1 = 0
    count4 = 0
    for sequence, target in generate_sequences(50):
      pred1 = s.predict_from_sequence(sequence, beam_width=1)
      pred4 = s.predict_from_sequence(sequence, beam_width=4)
      if pred1 == sequence:
        count1 += 1
      if pred4 == sequence:
        count4 += 1
      embedding = s.predict_embedding(sequence)
      assert pred1 == s.predict_from_embedding(embedding, beam_width=1)
      assert pred4 == s.predict_from_embedding(embedding, beam_width=4)

    # Check that it got at least a quarter of them correct.

    assert count1 >= 12
    assert count4 >= 12
