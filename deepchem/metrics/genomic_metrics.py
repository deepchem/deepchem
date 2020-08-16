"""Evaluation Metrics for Genomics Datasets."""

from typing import List, Optional
import numpy as np
from scipy.signal import correlate2d

from deepchem.models import Model
from deepchem.data import NumpyDataset


def get_motif_scores(encoded_sequences: np.ndarray,
                     motif_names: List[str],
                     max_scores: Optional[int] = None,
                     return_positions: bool = False,
                     GC_fraction: float = 0.4) -> np.ndarray:
  """Computes pwm log odds.

  Parameters
  ----------
  encoded_sequences: np.ndarray
    A numpy array of shape `(N_sequences, N_letters, sequence_length, 1)`.
  motif_names: List[str]
    List of motif file names.
  max_scores: int, optional
    Get top `max_scores` scores.
  return_positions: bool, default False
    Whether to return postions or not.
  GC_fraction: float, default 0.4
    GC fraction in background sequence.

  Returns
  -------
  np.ndarray
    A numpy array of complete score. The shape is `(N_sequences, num_motifs, seq_length)` by default.
    If max_scores, the shape of score array is `(N_sequences, num_motifs*max_scores)`.
    If max_scores and return_positions, the shape of score array with max scores and their positions.
    is `(N_sequences, 2*num_motifs*max_scores)`.

  Notes
  -----
  This method requires simdna to be installed.
  """
  try:
    import simdna
    from simdna import synthetic
  except ModuleNotFoundError:
    raise ValueError("This function requires simdna to be installed.")

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


def get_pssm_scores(encoded_sequences: np.ndarray,
                    pssm: np.ndarray) -> np.ndarray:
  """
  Convolves pssm and its reverse complement with encoded sequences
  and returns the maximum score at each position of each sequence.

  Parameters
  ----------
  encoded_sequences: np.ndarray
    A numpy array of shape `(N_sequences, N_letters, sequence_length, 1)`.
  pssm: np.ndarray
    A numpy array of shape `(4, pssm_length)`.

  Returns
  -------
  scores: np.ndarray
    A numpy array of shape `(N_sequences, sequence_length)`.
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


def in_silico_mutagenesis(model: Model,
                          encoded_sequences: np.ndarray) -> np.ndarray:
  """Computes in-silico-mutagenesis scores

  Parameters
  ----------
  model: Model
    This can be any model that accepts inputs of the required shape and produces
    an output of shape `(N_sequences, N_tasks)`.
  encoded_sequences: np.ndarray
    A numpy array of shape `(N_sequences, N_letters, sequence_length, 1)`

  Returns
  -------
  np.ndarray
    A numpy array of ISM scores. The shape is `(num_task, N_sequences, N_letters, sequence_length, 1)`.
  """
  # Shape (N_sequences, num_tasks)
  wild_type_predictions = model.predict(NumpyDataset(encoded_sequences))
  # check whether wild_type_predictions is np.ndarray or not
  assert isinstance(wild_type_predictions, np.ndarray)
  num_tasks = wild_type_predictions.shape[1]
  # Shape (N_sequences, N_letters, sequence_length, 1, num_tasks)
  mutagenesis_scores = np.empty(
      encoded_sequences.shape + (num_tasks,), dtype=np.float32)
  # Shape (N_sequences, num_tasks, 1, 1, 1)
  wild_type_predictions = wild_type_predictions[:, np.newaxis, np.newaxis,
                                                np.newaxis]
  for sequence_index, (sequence, wild_type_prediction) in enumerate(
      zip(encoded_sequences, wild_type_predictions)):

    # Mutates every position of the sequence to every letter
    # Shape (N_letters * sequence_length, N_letters, sequence_length, 1)
    # Breakdown:
    # Shape of sequence[np.newaxis] (1, N_letters, sequence_length, 1)
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
    # check whether wild_type_predictions is np.ndarray or not
    assert isinstance(mutated_predictions, np.ndarray)
    mutated_predictions = mutated_predictions.reshape(sequence.shape +
                                                      (num_tasks,))
    mutagenesis_scores[
        sequence_index] = wild_type_prediction - mutated_predictions
  rolled_scores = np.rollaxis(mutagenesis_scores, -1)
  return rolled_scores
