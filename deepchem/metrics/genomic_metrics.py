"""Evaluation Metrics for Genomics Datasets."""

import numpy as np
from deepchem.data import NumpyDataset
from scipy.signal import correlate2d


def get_motif_scores(encoded_sequences,
                     motif_names,
                     max_scores=None,
                     return_positions=False,
                     GC_fraction=0.4):
  """Computes pwm log odds.

  Parameters
  ----------
  encoded_sequences: np.ndarray
    A 4darray of shape `(N_sequences, N_letters, sequence_length, 1)`
    array
  motif_names: list[str]
    List of strings with motif names.
  max_scores: int, optional
    The maximum score to allow
  return_positions: bool, optional
    TODO
  GC_fraction : float, optional
    TODO

  Returns
  -------
  (N_sequences, num_motifs, seq_length) complete score array by default.
  If max_scores, (N_sequences, num_motifs*max_scores) max score array.
  If max_scores and return_positions, (N_sequences, 2*num_motifs*max_scores)
  array with max scores and their positions.
  """
  import simdna
  from simdna import synthetic
  loaded_motifs = synthetic.LoadedEncodeMotifs(
      simdna.ENCODE_MOTIFS_PATH, pseudocountProb=0.001)
  num_samples, _, seq_length, _ = encoded_sequences.shape
  scores = np.ones((num_samples, len(motif_names), seq_length))
  for j, motif_name in enumerate(motif_names):
    pwm = loaded_motifs.getPwm(motif_name).getRows().T
    log_pwm = np.log(pwm)
    gc_pwm = 0.5 * np.array(
        [[1 - GC_fraction, GC_fraction, GC_fraction, 1 - GC_fraction]] * len(
            pwm[0])).T
    gc_log_pwm = np.log(gc_pwm)
    log_scores = get_pssm_scores(encoded_sequences, log_pwm)
    gc_log_scores = get_pssm_scores(encoded_sequences, gc_log_pwm)
    scores[:, j, :] = log_scores - gc_log_scores
  if max_scores is not None:
    sorted_scores = np.sort(scores)[:, :, ::-1][:, :, :max_scores]
    if return_positions:
      sorted_positions = scores.argsort()[:, :, ::-1][:, :, :max_scores]
      return np.concatenate(
          (sorted_scores.reshape((num_samples, len(motif_names) * max_scores)),
           sorted_positions.reshape(
               (num_samples, len(motif_names) * max_scores))),
          axis=1)
    else:
      return sorted_scores.reshape((num_samples, len(motif_names) * max_scores))
  else:
    return scores


def get_pssm_scores(encoded_sequences, pssm):
  """
  Convolves pssm and its reverse complement with encoded sequences
  and returns the maximum score at each position of each sequence.

  Parameters
  ----------
  encoded_sequences: 3darray
       (N_sequences, N_letters, sequence_length, 1) array
  pssm: 2darray
      (4, pssm_length) array

  Returns
  -------
  scores: 2darray
      (N_sequences, sequence_length)
  """
  encoded_sequences = encoded_sequences.squeeze(axis=3)
  # initialize fwd and reverse scores to -infinity
  fwd_scores = np.full_like(encoded_sequences, -np.inf, float)
  rc_scores = np.full_like(encoded_sequences, -np.inf, float)
  # cross-correlate separately for each base,
  # for both the PSSM and its reverse complement
  for base_indx in range(encoded_sequences.shape[1]):
    base_pssm = pssm[base_indx][None]
    base_pssm_rc = base_pssm[:, ::-1]
    fwd_scores[:, base_indx, :] = correlate2d(
        encoded_sequences[:, base_indx, :], base_pssm, mode='same')
    rc_scores[:, base_indx, :] = correlate2d(
        encoded_sequences[:, -(base_indx + 1), :], base_pssm_rc, mode='same')
  # sum over the bases
  fwd_scores = fwd_scores.sum(axis=1)
  rc_scores = rc_scores.sum(axis=1)
  # take max of fwd and reverse scores at each position
  scores = np.maximum(fwd_scores, rc_scores)
  return scores


def in_silico_mutagenesis(model, X):
  """Computes in-silico-mutagenesis scores

  Parameters
  ----------
  model: Model
    This can be any model that accepts inputs of the required shape and produces
    an output of shape (N_sequences, N_tasks).
  X: ndarray
    Shape (N_sequences, N_letters, sequence_length, 1)

  Returns
  -------
  (num_task, N_sequences, N_letters, sequence_length, 1) ISM score array.
  """
  # Shape (N_sequences, num_tasks)
  wild_type_predictions = model.predict(NumpyDataset(X))
  num_tasks = wild_type_predictions.shape[1]
  #Shape (N_sequences, N_letters, sequence_length, 1, num_tasks)
  mutagenesis_scores = np.empty(X.shape + (num_tasks,), dtype=np.float32)
  # Shape (N_sequences, num_tasks, 1, 1, 1)
  wild_type_predictions = wild_type_predictions[:, np.newaxis, np.newaxis,
                                                np.newaxis]
  for sequence_index, (sequence, wild_type_prediction) in enumerate(
      zip(X, wild_type_predictions)):

    # Mutates every position of the sequence to every letter
    # Shape (N_letters * sequence_length, N_letters, sequence_length, 1)
    # Breakdown:
    #  Shape of sequence[np.newaxis] (1, N_letters, sequence_length, 1)
    mutated_sequences = np.repeat(
        sequence[np.newaxis], np.prod(sequence.shape), axis=0)

    # remove wild-type
    # len(arange) = N_letters * sequence_length
    arange = np.arange(len(mutated_sequences))
    # len(horizontal cycle) = N_letters * sequence_length
    horizontal_cycle = np.tile(np.arange(sequence.shape[1]), sequence.shape[0])
    mutated_sequences[arange, :, horizontal_cycle, :] = 0

    # add mutant
    vertical_repeat = np.repeat(np.arange(sequence.shape[0]), sequence.shape[1])
    mutated_sequences[arange, vertical_repeat, horizontal_cycle, :] = 1
    # make mutant predictions
    mutated_predictions = model.predict(NumpyDataset(mutated_sequences))
    mutated_predictions = mutated_predictions.reshape(sequence.shape +
                                                      (num_tasks,))
    mutagenesis_scores[
        sequence_index] = wild_type_prediction - mutated_predictions
  rolled_scores = np.rollaxis(mutagenesis_scores, -1)
  return rolled_scores
