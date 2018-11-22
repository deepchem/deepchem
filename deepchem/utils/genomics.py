"""
Genomic data handling utilities.
"""
import numpy as np


def seq_one_hot_encode(sequences, letters='ATCGN'):
  """One hot encodes list of genomic sequences.

  Sequences encoded have shape (N_sequences, N_letters, sequence_length, 1).
  These sequences will be processed as images with one color channel.

  Parameters
  ----------
  sequences: np.ndarray
    Array of genetic sequences
  letters: str
    String with the set of possible letters in the sequences.

  Raises
  ------
  ValueError:
    If sequences are of different lengths.

  Returns
  -------
  np.ndarray: Shape (N_sequences, N_letters, sequence_length, 1).
  """

  # The label encoder is given characters for ACGTN
  letter_encoder = {l: i for i, l in enumerate(letters)}
  alphabet_length = len(letter_encoder)

  # Peak at the first sequence to get the length of the sequence.
  try:
    first_seq = next(sequences)
    tail_seq = sequences
  except TypeError:
    first_seq = sequences[0]
    tail_seq = sequences[1:]

  sequence_length = len(first_seq)

  seqs = []

  seqs.append(
      _seq_to_encoded(first_seq, letter_encoder, alphabet_length,
                      sequence_length))

  for other_seq in tail_seq:
    if len(other_seq) != sequence_length:
      raise ValueError

    seqs.append(
        _seq_to_encoded(other_seq, letter_encoder, alphabet_length,
                        sequence_length))

  return np.expand_dims(np.array(seqs), -1)


def _seq_to_encoded(seq, letter_encoder, alphabet_length, sequence_length):
  b = np.zeros((alphabet_length, sequence_length))
  seq_ints = [letter_encoder[s] for s in seq]
  b[seq_ints, np.arange(sequence_length)] = 1

  return b


def encode_fasta_sequence(fname):
  """
  Loads fasta file and returns an array of one-hot sequences.

  Parameters
  ----------
  fname: str
    Filename of fasta file.

  Returns
  -------
  np.ndarray: Shape (N_sequences, 5, sequence_length, 1).
  """

  return encode_bio_sequence(fname)


def encode_bio_sequence(fname, file_type="fasta", letters="ATCGN"):
  """
  Loads a sequence file and returns an array of one-hot sequences.

  Parameters
  ----------
  fname: str
    Filename of fasta file.
  file_type: str
    The type of file encoding to process, e.g. fasta or fastq, this
    is passed to Biopython.SeqIO.parse.
  letters: str
    The set of letters that the sequences consist of, e.g. ATCG.

  Returns
  -------
  np.ndarray: Shape (N_sequences, N_letters, sequence_length, 1).
  """

  from Bio import SeqIO

  sequences = SeqIO.parse(fname, file_type)
  return seq_one_hot_encode(sequences, letters)
