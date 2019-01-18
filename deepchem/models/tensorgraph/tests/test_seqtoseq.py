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

    sequence_length = 8
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

  def test_aspuru_guzik(self):
    """Test that the aspuru_guzik encoder doesn't hard error.
    This model takes too long to fit to do an overfit test
    """
    train_smiles = [
        'Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C',
        'Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1',
        'COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O)oc1ccccc12',
        'O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1ccccc1',
        'NC(=O)NC(Cc1ccccc1)C(=O)O', 'CCn1c(CSc2nccn2C)nc2cc(C(=O)O)ccc21',
        'CCc1cccc2c1NC(=O)C21C2C(=O)N(Cc3ccccc3)C(=O)C2C2CCCN21',
        'COc1ccc(C2C(C(=O)NCc3ccccc3)=C(C)N=C3N=CNN32)cc1OC',
        'CCCc1cc(=O)nc(SCC(=O)N(CC(C)C)C2CCS(=O)(=O)C2)[nH]1',
        'CCn1cnc2c1c(=O)n(CC(=O)Nc1cc(C)on1)c(=O)n2Cc1ccccc1'
    ]
    tokens = set()
    for s in train_smiles:
      tokens = tokens.union(set(c for c in s))
    tokens = sorted(list(tokens))
    max_length = max(len(s) for s in train_smiles) + 1
    s = dc.models.tensorgraph.models.seqtoseq.AspuruGuzikAutoEncoder(
        tokens, max_length)

    def generate_sequences(smiles, epochs):
      for i in range(epochs):
        for s in smiles:
          yield (s, s)

    s.fit_sequences(generate_sequences(train_smiles, 100))

    # Test it out.
    pred1 = s.predict_from_sequences(train_smiles, beam_width=1)
    pred4 = s.predict_from_sequences(train_smiles, beam_width=4)
    embeddings = s.predict_embeddings(train_smiles)
    pred1e = s.predict_from_embeddings(embeddings, beam_width=1)
    pred4e = s.predict_from_embeddings(embeddings, beam_width=4)
    for i in range(len(train_smiles)):
      assert pred1[i] == pred1e[i]
      assert pred4[i] == pred4e[i]

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
