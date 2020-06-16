"""
Callback functions that can be invoked while fitting a KerasModel.
"""

import tensorflow as tf
import sys
from deepchem.metrics import Metric


class StopIteration(Exception):

  def __init__(self, *args, **kwargs):
    Exception.__init__(*args, **kwargs)


class ValidationCallback(object):
  """Performs validation while training a KerasModel.

  This is a callback that can be passed to fit().  It periodically computes a
  set of metrics over a validation set and writes them to a file.  In addition,
  it can save the best model parameters found so far to a directory on disk,
  updating them every time it finds a new best validation score.

  If Tensorboard logging is enabled on the KerasModel, the metrics are also
  logged to Tensorboard.  This only happens when validation coincides with a
  step on which the model writes to the log.  You should therefore make sure
  that this callback's reporting interval is an even fraction or multiple of
  the model's logging interval.
  """

  def __init__(self,
               dataset,
               interval,
               metrics,
               output_file=sys.stdout,
               save_dir=None,
               save_metric=0,
               save_on_minimum=True,
               early_stop_metric=None,
               delta=0.01,
               patience=0):
    """Create a ValidationCallback.

    Parameters
    ----------
    dataset: dc.data.Dataset
      the validation set on which to compute the metrics
    interval: int
      the interval (in training steps) at which to perform validation
    metrics: list of dc.metrics.Metric
      metrics to compute on the validation set
    output_file: file
      to file to which results should be written
    save_dir: str
      if not None, the model parameters that produce the best validation score
      will be written to this directory
    save_metric: int
      the index of the metric to use when deciding whether to write a new set
      of parameters to disk
    save_on_minimum: bool
      if True, the best model is considered to be the one that minimizes the
      validation metric.  If False, the best model is considered to be the one
      that maximizes it.
    early_stop_metric: str/int, default None
      Indicates whether early stopping needs to be applied. If int, the index 
      of the metric to use when checking for early stopping. If str, the loss 
      on validation dataset is computed.
    delta: float, default 0.01
      Threshold for measuring the change in metric, to focus only on significant changes.
    patience: int, default 0,
      Number of steps with no improvement after which early stopping is enforced.
    """
    self.dataset = dataset
    self.interval = interval
    self.metrics = metrics
    self.output_file = output_file
    self.save_dir = save_dir
    self.save_metric = save_metric
    self.save_on_minimum = save_on_minimum
    self.early_stop_metric = early_stop_metric
    self.delta = delta  #Used only for early stopping
    self.patience = patience  #Used only for early stopping
    self.wait = 0  #Used only for early stopping
    self._best_score = None

  def __call__(self, model, step):
    """This is invoked by the KerasModel after every step of fitting.

    Parameters
    ----------
    model: KerasModel
      the model that is being trained
    step: int
      the index of the training step that has just completed
    """
    if step % self.interval != 0:
      return
    scores = model.evaluate(self.dataset, self.metrics)
    message = 'Step %d validation:' % step
    for key in scores:
      message += ' %s=%g' % (key, scores[key])
    print(message, file=self.output_file)
    if model.tensorboard:
      for key in scores:
        model._log_value_to_tensorboard(tag=key, simple_value=scores[key])

    if self.early_stopping_metric is not None:
      if self.early_stopping_metric == 'loss':
        early_stop_score = model.compute_loss(self.dataset, transformers=[])
        model._log_value_to_tensorboard(
            tag='loss', sample_value=early_stop_score)
      else:
        early_stop_score = scores[self.metrics[self.early_stopping_metric].name]

      if not self.save_on_minimum:
        early_stop_score = -early_stop_score
      if self._best_score is None or (self._best_score - early_stop_score >
                                      self.delta):
        self.wait = 0  #Reset counter if improvement recorded
        self._best_score = early_stop_score
        if self.save_dir is not None:
          model.save_checkpoint(model_dir=self.save_dir)
      else:
        self.wait += 1
        if self.wait >= self.patience:
          raise StopIteration("No improvement in metric value. \
                              Enforcing early stopping.")
    else:
      if self.save_dir is not None:
        score = scores[self.metrics[self.save_metric].name]
        if not self.save_on_minimum:
          score = -score
        if self._best_score is None or score < self._best_score:
          model.save_checkpoint(model_dir=self.save_dir)
          self._best_score = score
