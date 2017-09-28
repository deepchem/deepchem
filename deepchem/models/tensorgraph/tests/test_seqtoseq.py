import deepchem as dc
import numpy as np
import unittest


def generate_sequences(sequence_length, num_sequences):
  for i in range(num_sequences):
    seq = [
        np.random.randint(10)
        for x in range(np.random.randint(1, sequence_length + 1))
    ]
    yield (seq, seq)


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

    s.fit_sequences(generate_sequences(sequence_length, 25000))

    # Test it out.

    tests = [seq for seq, target in generate_sequences(sequence_length, 50)]
    pred1 = s.predict_from_sequences(tests, beam_width=1)
    pred4 = s.predict_from_sequences(tests, beam_width=4)
    embeddings = s.predict_embeddings(tests)
    pred1e = s.predict_from_embeddings(embeddings, beam_width=1)
    pred4e = s.predict_from_embeddings(embeddings, beam_width=4)
    count1 = 0
    count4 = 0
    for i in range(len(tests)):
      if pred1[i] == tests[i]:
        count1 += 1
      if pred4[i] == tests[i]:
        count4 += 1
      assert pred1[i] == pred1e[i]
      assert pred4[i] == pred4e[i]

    # Check that it got at least a quarter of them correct.

    assert count1 >= 12
    assert count4 >= 12

  def test_variational(self):
    """Test using a SeqToSeq model as a variational autoenconder."""

    sequence_length = 10
    tokens = list(range(10))
    s = dc.models.SeqToSeq(
        tokens,
        tokens,
        sequence_length,
        encoder_layers=2,
        decoder_layers=2,
        embedding_dimension=128,
        learning_rate=0.01)

    # Actually training a VAE takes far too long for a unit test.  Just run a
    # few steps of training to make sure nothing crashes, then check that the
    # results are at least internally consistent.

    s.fit_sequences(generate_sequences(sequence_length, 1000))
    for sequence, target in generate_sequences(sequence_length, 10):
      pred1 = s.predict_from_sequences([sequence], beam_width=1)
      embedding = s.predict_embeddings([sequence])
      assert pred1 == s.predict_from_embeddings(embedding, beam_width=1)
