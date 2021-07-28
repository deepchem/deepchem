from typing import Optional, Union, Dict, List
import tensorflow as tf
import torch
import numpy as np
numeric = Union[tf.Tensor, torch.Tensor, int, float, complex, np.number]
tensor = Union[tf.Tensor, torch.Tensor]

class Logger(object):
    def __init__(self):
        raise NotImplementedError

    def setup(self, config: Dict):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

    def log_batch(self, loss: Dict, step: int, inputs: tensor, labels: tensor, location: Optional[str] = None):
        raise NotImplementedError

    def log_epoch(self, data: Dict, epoch: int, location: Optional[str] = None):
        raise NotImplementedError

    def log_values(self, values: Dict, step: int, location: Optional[str] = None):
        raise NotImplementedError

    def end_run(self, data: Dict, group: Optional[str] = None):
        raise NotImplementedError

