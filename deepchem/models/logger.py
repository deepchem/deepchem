from typing import Optional, Union, Dict
import tensorflow as tf
import torch
import numpy as np
from deepchem.models import Model

numeric = Union[tf.Tensor, torch.Tensor, int, float, complex, np.number]
tensor = Union[tf.Tensor, torch.Tensor]


class Logger(object):
  """
  Abstract base class for loggers used in DeepChem.
  """

  def __init__(self):
    """Abstract class for all loggers.

    This is intended only for convenience of subclass implementations
    and should not be invoked directly. Individual loggers should provide
    their own implementation for methods.
    """
    raise NotImplementedError

  def __iter__(self):
    """
    Allow list of loggers to be iterable.
    """
    return self

  def setup(self, config: Dict):
    """
    Set up and initialize a logger.

    Parameters
    ----------
    config: Dict
      configuration/settings to be passed to logger.
    """
    raise NotImplementedError

  def finish(self):
    """
    Close a logger and end its process.
    """
    raise NotImplementedError

  def log_batch(self,
                loss: Dict,
                step: int,
                inputs: tensor,
                labels: tensor,
                location: Optional[str] = None):
    """Log values for a single training batch.

    Parameters
    ----------
    loss: Dict
      the loss values for the batch
    step: int
      the current training step
    inputs: tensor
      batch input tensor
    labels: tensor
      batch labels tensor
    location: str, optional (default None)
      the logging location or chart panel section/group
    """
    raise NotImplementedError

  def log_epoch(self, data: Dict, epoch: int, location: Optional[str] = None):
    """Log values for a epoch.

    Parameters
    ----------
    data: Dict
      data values to be logged
    epoch: int
      epoch number
    location: str, optional (default None)
      the logging location or chart panel section/group
    """
    raise NotImplementedError

  def log_values(self, data: Dict, step: int, location: Optional[str] = None):
    """Log values for a certain step in training/evaluation.

    Parameters
    ----------
    data: Dict
      data values to be logged
    step: int
      epoch number
    location: str, optional (default None)
      the logging location or chart panel section/group
    """
    raise NotImplementedError

  def on_fit_end(self, data: Dict):
    """Called before the end of training.

    Parameters
    ----------
    data: Dict
      Training summary values to be logged
    """
    raise NotImplementedError

  def save_checkpoint(self,
                      model_dir: str,
                      dc_model: Model,
                      checkpoint_name: str,
                      value_name: str,
                      value: numeric,
                      max_checkpoints_to_track: int,
                      checkpoint_on_min: bool,
                      metadata: Optional[Dict] = None):
    """Save model checkpoint.

    Parameters
    ----------
    model_dir: str
      directory containing model checkpoints
    dc_model: Model
      DeepChem model object to be saved
    checkpoint_name: str
      name of the checkpoint
    value_name: str
      the name metric to checkpoint on
    value: numeric
      the value of the metric to checkpoint on
    max_checkpoints_to_track: int
      the maximum number of checkpoints to track. Logger implementations
      can choose what to do with older checkpoints (discard, keep, tag as old, etc.)
    checkpoint_on_min:
      if True, the best model is considered to be the one that minimizes the
      value. If False, the best model is considered to be the one
      that maximizes it.
    metadata: Dict, optional(default None)
      metadata to be save along with the checkpoint
    """
    raise NotImplementedError
