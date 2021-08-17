from typing import Optional, Union, Dict
import tensorflow as tf
import torch
import numpy as np
from deepchem.models import Model

numeric = Union[tf.Tensor, torch.Tensor, int, float, complex, np.number]
tensor = Union[tf.Tensor, torch.Tensor]


class Logger(object):
  """
  Abstract base class for loggers.
  """

  def __init__(self):
    """Abstract class for all loggers.

    This is intended only for convenience of subclass implementations
    and should not be invoked directly. Individual loggers should provide
    their own implementation for methods.
    """
    raise NotImplementedError

  def __iter__(self):
    return self

  def setup(self, config: Dict):
    raise NotImplementedError

  def finish(self):
    raise NotImplementedError

  def log_batch(self,
                loss: Dict,
                step: int,
                inputs: tensor,
                labels: tensor,
                location: Optional[str] = None):
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
