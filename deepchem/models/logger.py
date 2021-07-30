from typing import Optional, Union, Dict, List
import tensorflow as tf
import torch
import numpy as np
from deepchem.models import Model

numeric = Union[tf.Tensor, torch.Tensor, int, float, complex, np.number]
tensor = Union[tf.Tensor, torch.Tensor]


class Logger(object):

  def __init__(self):
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
                location: Optional[str] = None,
                model: Optional[Model] = None,
                checkpoint_metric: Optional[str] = None,
                checkpoint_metric_value: Optional[numeric] = None,
                checkpoint_on_min: bool = True):
    raise NotImplementedError

  def log_epoch(self,
                data: Dict,
                epoch: int,
                location: Optional[str] = None,
                model: Optional[Model] = None,
                checkpoint_metric: Optional[str] = None,
                checkpoint_metric_value: Optional[numeric] = None,
                checkpoint_on_min: Optional[bool] = True):
    raise NotImplementedError

  def log_values(self,
                 data: Dict,
                 step: int,
                 location: Optional[str] = None,
                 model: Optional[Model] = None,
                 checkpoint_metric: Optional[str] = None,
                 checkpoint_metric_value: Optional[numeric] = None,
                 checkpoint_on_min: bool = True):
    raise NotImplementedError

  def on_fit_end(self,
                 data: Dict,
                 location: Optional[str] = None,
                 model: Optional[Model] = None,
                 checkpoint_metric: Optional[str] = None,
                 checkpoint_metric_value: Optional[numeric] = None,
                 checkpoint_on_min: bool = True):
    raise NotImplementedError
