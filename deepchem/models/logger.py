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
    config: Configuration/settings to be passed to logger.
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
    """
    Log a single training batch.

    Parameters
    ----------
    loss
    step
    inputs
    labels
    location
    """
    raise NotImplementedError

  def log_epoch(self,
                data: Dict,
                epoch: int,
                location: Optional[str] = None):
    raise NotImplementedError

  def log_values(self,
                 data: Dict,
                 step: int,
                 location: Optional[str] = None):
    raise NotImplementedError

  def on_fit_end(self,
                 data: Dict):
    raise NotImplementedError

  def save_checkpoint(self,
                      path: str,
                      dc_model: Model,
                      checkpoint_name: str,
                      value_name: str,
                      value: numeric,
                      max_checkpoints_to_track: int,
                      checkpoint_on_min: bool,
                      metadata: Optional[Dict] = None):
    raise NotImplementedError
