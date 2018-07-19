"""Evaluation Metrics for Genomics Datasets."""

import numpy as np
from deepchem.utils.genomics import loaded_motifs


def get_motif_scores(encoded_sequences,
                     motif_names,
                     max_scores=None,
                     return_positions=False,
                     GC_fraction=0.4):
  """Computes pwm log odds.

  Parameters
  ----------
  encoded_sequences : 4darray
  motif_names : list of strings
  max_scores : int, optional
  return_positions : boolean, optional
  GC_fraction : float, optional

  Returns
  -------
  (num_samples, num_motifs, seq_length) complete score array by default.
  If max_scores, (num_samples, num_motifs*max_scores) max score array.
  If max_scores and return_positions, (num_samples, 2*num_motifs*max_scores)
  array with max scores and their positions.
  """
  num_samples, _, _, seq_length = encoded_sequences.shape
  scores = np.ones((num_samples, len(motif_names), seq_length))
  for j, motif_name in enumerate(motif_names):
    pwm = loaded_motifs.getPwm(motif_name).getRows().T
    log_pwm = np.log(pwm)
    gc_pwm = 0.5 * np.array(
        [[1 - GC_fraction, GC_fraction, GC_fraction, 1 - GC_fraction]] * len(
            pwm[0])).T
    gc_log_pwm = np.log(gc_pwm)
    scores[:, j, :] = get_pssm_scores(encoded_sequences,
                                      log_pwm) - get_pssm_scores(
                                          encoded_sequences, gc_log_pwm)
  if max_scores is not None:
    sorted_scores = np.sort(scores)[:, :, ::-1][:, :, :max_scores]
    if return_positions:
      sorted_positions = scores.argsort()[:, :, ::-1][:, :, :max_scores]
      return np.concatenate(
          (sorted_scores.reshape((num_samples, len(motif_names) * max_scores)),
           sorted_positions.reshape((num_samples,
                                     len(motif_names) * max_scores))),
          axis=1)
    else:
      return sorted_scores.reshape((num_samples, len(motif_names) * max_scores))
  else:
    return scores


def get_pssm_scores(encoded_sequences, pssm):
  """Maximum PSSM scores.

  Convolves pssm and its reverse complement with encoded sequences
  and returns the maximum score at each position of each sequence.

  Parameters
  ----------
  encoded_sequences: 3darray
       (num_examples, 1, 4, seq_length) array
  pssm: 2darray
      (4, pssm_length) array

  Returns
  -------
  scores: 2darray
      (num_examples, seq_length) array
  """
  encoded_sequences = encoded_sequences.squeeze(axis=1)
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


def get_pssm_scores(encoded_sequences, pssm):
  """
  Convolves pssm and its reverse complement with encoded sequences
  and returns the maximum score at each position of each sequence.

  Parameters
  ----------
  encoded_sequences: 3darray
       #(num_examples, 1, 4, seq_length) array
       (N_sequences, N_letters, sequence_length, 1) array
  pssm: 2darray
      (4, pssm_length) array

  Returns
  -------
  scores: 2darray
      (num_examples, seq_length) array
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
