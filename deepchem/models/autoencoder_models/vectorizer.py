import numpy as np
import itertools
import random
from rdkit import Chem


class CharacterTable(object):
  '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    first version by rmcgibbo
    '''

  def __init__(self, chars, maxlen):
    self.chars = sorted(set(chars))
    self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
    self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    self.maxlen = maxlen

  def encode(self, C, maxlen=None):
    maxlen = maxlen if maxlen else self.maxlen
    X = np.zeros((maxlen, len(self.chars)))
    for i, c in enumerate(C):
      X[i, self.char_indices[c]] = 1
    return X

  def decode(self, X, mode='argmax'):
    if mode == 'argmax':
      X = X.argmax(axis=-1)
    elif mode == 'choice':
      X = np.apply_along_axis(lambda vec: \
                                  np.random.choice(len(vec), 1,
                                                   p=(vec / np.sum(vec))),
                              axis=-1, arr=X).ravel()
    return str.join('', (self.indices_char[x] for x in X))


class SmilesDataGenerator(object):
  """
    Given a list of SMILES strings,
    returns a generator that returns batches of
    randomly sampled strings encoded as one-hot,
    as well as a weighting vector indicating true length of string
    """
  SMILES_CHARS = [
      ' ', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5',
      '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M',
      'N', 'O', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '[', '\\', ']', 'a', 'b',
      'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u'
  ]

  def __init__(self,
               words,
               maxlen,
               pad_char=' ',
               pad_min=1,
               pad_weight=0.0,
               test_split=0.20):
    self.maxlen = maxlen
    self.words = words
    self.max_words = len(words)
    self.word_ixs = range(self.max_words)
    self.shuffled_word_ixs = range(self.max_words)
    random.shuffle(self.shuffled_word_ixs)

    self.pad_char = pad_char
    self.pad_min = pad_min
    self.pad_weight = pad_weight
    self.test_split = test_split
    self.chars = sorted(
        set.union(set(SmilesDataGenerator.SMILES_CHARS), set(pad_char)))
    self.table = CharacterTable(self.chars, self.maxlen)

  def encode(self, word):
    padded_word = word + self.pad_char * (self.maxlen - len(word))
    return self.table.encode(padded_word)

  def weight(self, word):
    weight_vec = np.ones((self.maxlen,)) * self.pad_weight
    weight_vec[np.arange(min(len(word) + self.pad_min, self.maxlen))] = 1
    return weight_vec

  def sample(self, predicate=None):
    if predicate:
      word_ix = random.choice(self.word_ixs)
      if not predicate(word_ix):
        return self.sample(predicate=predicate)
      word = self.words[self.shuffled_word_ixs[word_ix]]
    else:
      word = random.choice(self.words)
    if len(word) < self.maxlen:
      return word
    return self.sample(predicate=predicate)

  def train_sample(self):
    if self.test_split > 0:
      threshold = self.max_words * self.test_split
      return self.sample(lambda word_ix: word_ix >= threshold)
    return self.sample()

  def test_sample(self):
    if self.test_split > 0:
      threshold = self.max_words * self.test_split
      return self.sample(lambda word_ix: word_ix < threshold)
    return self.sample()

  def generator(self, batch_size, sample_func=None):
    while True:
      data_tensor = np.zeros(
          (batch_size, self.maxlen, len(self.chars)), dtype=np.bool)
      weight_tensor = np.zeros((batch_size, self.maxlen))
      for word_ix in range(batch_size):
        if not sample_func:
          sample_func = self.sample
        word = sample_func()
        data_tensor[word_ix, ...] = self.encode(word)
        weight_tensor[word_ix, ...] = self.weight(word)
      yield (data_tensor, data_tensor, weight_tensor)

  def train_generator(self, batch_size):
    return self.generator(batch_size, sample_func=self.train_sample)

  def test_generator(self, batch_size):
    return self.generator(batch_size, sample_func=self.test_sample)


class CanonicalSmilesDataGenerator(SmilesDataGenerator):
  """
    Given a list of SMILES strings,
    returns a generator that returns batches of
    randomly sampled strings, canonicalized, encoded as one-hot,
    as well as a weighting vector indicating true length of string
    """

  def sample(self, predicate=None):
    mol = Chem.MolFromSmiles(
        super(CanonicalSmilesDataGenerator, self).sample(predicate=predicate))
    if mol:
      canon_word = Chem.MolToSmiles(mol)
      if len(canon_word) < self.maxlen:
        return canon_word
    return self.sample(predicate=predicate)

  def train_sample(self):
    if self.test_split > 0:
      threshold = self.max_words * self.test_split
      return self.sample(lambda word_ix: word_ix >= threshold)
    return self.sample()

  def test_sample(self):
    if self.test_split > 0:
      threshold = self.max_words * self.test_split
      return self.sample(lambda word_ix: word_ix < threshold)
    return self.sample()
