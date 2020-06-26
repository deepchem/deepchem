"""
Utility functions to evaluate models on datasets.
"""
import csv
import logging
import numpy as np
import warnings
import pandas as pd
import sklearn
from deepchem.trans import undo_transforms
from deepchem.metrics import from_one_hot
from deepchem.metrics import Metric

logger = logging.getLogger(__name__)


def _process_metric_input(metrics):
  """A private helper method which processes metrics correctly.

  Metrics can be input as `dc.metrics.Metric` objects, lists of
  `dc.metrics.Metric` objects, or as raw metric functions or lists of
  raw metric functions. Metric functions are functions which accept
  two arguments `y_true, y_pred` both of which must be `np.ndarray`
  objects and return a float value. This functions normalizes these
  different types of inputs to type `list[dc.metrics.Metric]` object
  for ease of later processing.

  Note that raw metric functions which don't have names attached will
  simply be named "metric-#" where # is their position in the provided
  metric list. For example, "metric-1" or "metric-7"

  Parameters
  ----------
  metrics: dc.metrics.Metric/list[dc.metrics.Metric]/metric function/ list[metric function]
    Input metrics to process.

  Returns
  -------
  final_metrics: list[dc.metrics.Metric]
    Converts all input metrics and outputs a list of
    `dc.metrics.Metric` objects.
  """
  # Make sure input is a list
  if not len(metrics):
    metrics = [metrics]
  final_metrics = []
  for i, metric in enumerate(metrics):
    # Ensure that metric is wrapped in a list.
    if isinstance(metric, Metric):
      final_metrics.append(metric)
    # This case checks if input is a function then wraps a
    # dc.metrics.Metric object around it
    elif callable(metric):
      wrap_metric = Metric(metric, name="metric-%d" % i)
      final_metrics.append(wrap_metric)
    else:
      raise ValueError(
          "metrics must be one of metric function / dc.metrics.Metric object / list of dc.metrics.Metric or metric functions."
      )
  return final_metrics


def relative_difference(x, y):
  """Compute the relative difference between x and y

  The two argument arrays must have the same shape.

  Parameters
  ----------
  x: np.ndarray
    First input array
  y: np.ndarray
    Second input array

  Returns
  -------
  z: np.ndarray
    We will have `z == np.abs(x-y) / np.abs(max(x, y))`.
  """
  z = np.abs(x - y) / np.abs(max(x, y))
  return z


def threshold_predictions(y, threshold=0.5):
  """Threshold predictions from classification model.

  Parameters
  ----------
  y: np.ndarray
    Must have shape `(N, n_classes)` and be class probabilities.
  threshold: float, optional (Default 0.5)
    The threshold probability for the positive class.

  Returns
  -------
  y_out: np.ndarray
    Of shape `(N,)` with class predictions as integers ranging from 0
    to `n_classes-1`.
  """
  n_preds = len(y_pred)
  y_out = np.zeros_like(y)
  y_out = np.where(y_pred[:, 1] >= threshold, np.ones(n_preds),
                   np.zeros(n_preds))
  return y_out


class Evaluator(object):
  """Class that evaluates a model on a given dataset.

  The evaluator class is used to evaluate a `dc.models.Model` class on
  a given `dc.data.Dataset` object. The evaluator is aware of
  `dc.trans.Transformer` objects so will automatically undo any
  transformations which have been applied.

  Example
  -------
  >>> import numpy as np
  >>> X = np.random.rand(10, 5)
  >>> y = np.random.rand(10, 1)
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>> model = dc.models.MultitaskRegressor(1, 5)
  >>> transformers = []
  >>> evaluator = Evaluator(model, dataset, transformers)
  >>> metric = dc.metrics.Metric(dc.metrics.mae_score)
  >>> multitask_scores = evaluator.compute_model_performance([metric])
  """

  def __init__(self, model, dataset, transformers):
    self.model = model
    self.dataset = dataset
    self.output_transformers = [
        transformer for transformer in transformers if transformer.transform_y
    ]
    self.task_names = dataset.get_task_names()

  def output_statistics(self, scores, stats_out):
    """ Write computed stats to file.

    Parameters
    ----------
    scores: dict
      Dictionary mapping names of metrics to scores.
    stats_out: str
      Name of file to write scores to.
    """
    with open(stats_out, "w") as statsfile:
      statsfile.write(str(scores) + "\n")

  def output_predictions(self, y_preds, csv_out):
    """Writes predictions to file.

    Parameters
    ----------
    y_preds: np.ndarray
      Predictions to output
    csvfile: str
      Name of file to write predictions to.
    """
    mol_ids = self.dataset.ids
    n_tasks = len(self.task_names)
    y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
    assert len(y_preds) == len(mol_ids)
    with open(csv_out, "w") as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(["Compound"] + self.dataset.get_task_names())
      for mol_id, y_pred in zip(mol_ids, y_preds):
        csvwriter.writerow([mol_id] + list(y_pred))

  def compute_model_performance(self,
                                metrics,
                                csv_out=None,
                                stats_out=None,
                                per_task_metrics=False):
    """
    Computes statistics of model on test data and saves results to csv.

    Parameters
    ----------
    metrics: list
      List of dc.metrics.Metric objects
    csv_out: str, optional
      Filename to write CSV of model predictions.
    stats_out: str, optional
      Filename to write computed statistics.
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.

    Returns
    -------
    multitask_scores: dict
      Dictionary mapping names of metrics to metric scores.
    all_task_scores: dict, optional
      If `per_task_metrics == True`, then returns a second dictionary
      of scores for each task separately.
    """
    y = self.dataset.y
    y = undo_transforms(y, self.output_transformers)
    w = self.dataset.w

    if not len(metrics):
      return {}
    else:
      mode = metrics[0].mode
    y_pred = self.model.predict(self.dataset, self.output_transformers)
    if mode == "classification":
      y_pred_print = np.argmax(y_pred, -1)
    else:
      y_pred_print = y_pred
    multitask_scores = {}
    all_task_scores = {}

    if csv_out is not None:
      logger.info("Saving predictions to %s" % csv_out)
      self.output_predictions(y_pred_print, csv_out)

    # Compute multitask metrics
    for metric in metrics:
      if per_task_metrics:
        multitask_scores[metric.name], computed_metrics = metric.compute_metric(
            y, y_pred, w, per_task_metrics=True)
        all_task_scores[metric.name] = computed_metrics
      else:
        multitask_scores[metric.name] = metric.compute_metric(
            y, y_pred, w, per_task_metrics=False)

    if stats_out is not None:
      logger.info("Saving stats to %s" % stats_out)
      self.output_statistics(multitask_scores, stats_out)

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores


class GeneratorEvaluator(object):
  """Evaluate models on a stream of data.

  This class is a partner class to `Evaluator`. Instead of operating
  over datasets this class operates over a generator which yields
  batches of data to feed into provided model.

  Example
  -------
  >>> import numpy as np
  >>> X = np.random.rand(10, 5)
  >>> y = np.random.rand(10, 1)
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>> model = dc.models.MultitaskRegressor(1, 5)
  >>> transformers = []
  >>> generator = model.default_generator(dataset, pad_batches=False)
  >>> evaluator = Evaluator(model, generator, transformers)
  >>> multitask_scores = evaluator.compute_model_performance([metric])
  """

  def __init__(self, model, generator, transformers, labels=None, weights=None):
    """
    Parameters
    ----------
    model: Model
      Model to evaluate.
    generator: Generator
      Generator which yields batches to feed into the model. For a
      KerasModel, it should be a tuple of the form (inputs, labels,
      weights). The "correct" way to create this generator is to use
      `model.default_generator` as shown in the example above.
    transformers:
      Tranformers to "undo" when applied to the models outputs
    labels: list of Layer
      layers which are keys in the generator to compare to outputs
    weights: list of Layer
      layers which are keys in the generator for weight matrices
    """
    self.model = model
    self.generator = generator
    self.output_transformers = [
        transformer for transformer in transformers if transformer.transform_y
    ]
    self.label_keys = labels
    self.weights = weights
    if labels is not None and len(labels) != 1:
      raise ValueError("GeneratorEvaluator currently only supports one label")

  def compute_model_performance(self, metrics, per_task_metrics=False):
    """
    Computes statistics of model on test data and saves results to csv.

    Parameters
    ----------
    metrics: dc.metrics.Metric/list[dc.metrics.Metric]/function
      The set of metrics provided. This class attempts to do some
      intelligent handling of input. If a single `dc.metrics.Metric`
      object is provided or a list is provided, it will evaluate
      `self.model` on these metrics. If a function is provided, it is
      assumed to be a metric function that this method will attempt to
      wrap in a `dc.metrics.Metric` object. A metric function must
      accept two arguments, `y_true, y_pred` both of which are
      `np.ndarray` objects and return a floating point score.
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask
      dataset.

    Returns
    -------
    multitask_scores: dict
      Dictionary mapping names of metrics to metric scores.
    all_task_scores: dict, optional
      If `per_task_metrics == True`, then returns a second dictionary
      of scores for each task separately.
    """
    metrics = _process_metric_input(metrics)

    # We use y/w to aggregate labels/weights across generator.
    y = []
    w = []

    def generator_closure():
      if self.label_keys is None:
        weights = None
        # This is a KerasModel.
        for batch in self.generator:
          # Some datasets have weights
          try:
            inputs, labels, weights = batch
          except ValueError:
            try:
              inputs, labels, weights, ids = batch
            except ValueError:
              raise ValueError(
                  "Generator must yield values of form (input, labels, weights) or (input, labels, weights, ids)"
              )
          y.append(labels[0])
          if len(weights) > 0:
            w.append(weights[0])
          yield (inputs, labels, weights)

    # Process predictions and populate y/w lists
    y_pred = self.model.predict_on_generator(generator_closure())

    # Combine labels/weights
    y = np.concatenate(y, axis=0)
    w = np.concatenate(w, axis=0)

    multitask_scores = {}
    all_task_scores = {}

    # Undo data transformations.
    y = undo_transforms(y, self.output_transformers)
    y_pred = undo_transforms(y_pred, self.output_transformers)

    # Compute multitask metrics
    for metric in metrics:
      if per_task_metrics:
        multitask_scores[metric.name], computed_metrics = metric.compute_metric(
            y, y_pred, w, per_task_metrics=True)
        all_task_scores[metric.name] = computed_metrics
      else:
        multitask_scores[metric.name] = metric.compute_metric(
            y, y_pred, w, per_task_metrics=False)

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores
