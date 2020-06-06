"""
Utility functions to evaluate models on datasets.
"""
import csv
import numpy as np
import warnings
import pandas as pd
import sklearn
from deepchem.utils.save import log
from deepchem.trans import undo_transforms
from deepchem.metrics import from_one_hot

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"


def relative_difference(x, y):
  """Compute the relative difference between x and y"""
  return np.abs(x - y) / np.abs(max(x, y))


def threshold_predictions(y, threshold):
  y_out = np.zeros_like(y)
  for ind, pred in enumerate(y):
    y_out[ind] = 1 if pred > threshold else 0
  return y_out


# TODO(rbharath): This is now simple enough that we should probably get rid of
# Evaluator object to avoid clutter.
class Evaluator(object):
  """Class that evaluates a model on a given dataset."""

  def __init__(self, model, dataset, transformers, verbose=False):
    self.model = model
    self.dataset = dataset
    self.output_transformers = [
        transformer for transformer in transformers if transformer.transform_y
    ]
    self.task_names = dataset.get_task_names()
    self.verbose = verbose

  def output_statistics(self, scores, stats_out):
    """
    Write computed stats to file.
    """
    with open(stats_out, "w") as statsfile:
      statsfile.write(str(scores) + "\n")

  def output_predictions(self, y_preds, csv_out):
    """
    Writes predictions to file.

    Args:
      y_preds: np.ndarray
      csvfile: Open file object.
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
      log("Saving predictions to %s" % csv_out, self.verbose)
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
      log("Saving stats to %s" % stats_out, self.verbose)
      self.output_statistics(multitask_scores, stats_out)

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores


class GeneratorEvaluator(object):
  """
  Partner class to Evaluator.
  Instead of operating over datasets this class operates over Generator.
  Evaluate a Metric over a model and Generator.
  """

  def __init__(self, model, generator, transformers, labels=None, weights=None):
    """
    Parameters
    ----------
    model: Model
      Model to evaluate
    generator: Generator
      Generator which yields batches to feed into the model.  For a TensorGraph,
      each batch should be a dict mapping Layers to NumPy arrays.  For a
      KerasModel, it should be a tuple of the form (inputs, labels, weights).
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
    metrics: list
      List of dc.metrics.Metric objects
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.
    """
    y = []
    w = []

    def generator_closure():
      if self.label_keys is None:
        # This is a KerasModel.
        for batch in self.generator:
          inputs, labels, weights = batch
          y.append(labels[0])
          if len(weights) > 0:
            w.append(weights[0])
          yield batch
      else:
        # This is a TensorGraph.
        for feed_dict in self.generator:
          y.append(feed_dict[self.label_keys[0]])
          if len(self.weights) > 0:
            w.append(feed_dict[self.weights[0]])
          yield feed_dict

    if not len(metrics):
      return {}
    else:
      mode = metrics[0].mode
    y_pred = self.model.predict_on_generator(generator_closure())
    y = np.concatenate(y, axis=0)
    multitask_scores = {}
    all_task_scores = {}

    y = undo_transforms(y, self.output_transformers)
    y_pred = undo_transforms(y_pred, self.output_transformers)
    if len(w) != 0:
      w = np.array(w)
      if np.prod(w.shape) == y.shape[0]:
        w = np.reshape(w, newshape=(y.shape[0], 1))
      else:
        w = np.reshape(w, newshape=y.shape)

    # Compute multitask metrics
    n_classes = y.shape[-1]
    for metric in metrics:
      if per_task_metrics:
        multitask_scores[metric.name], computed_metrics = metric.compute_metric(
            y, y_pred, w, per_task_metrics=True, n_classes=n_classes)
        all_task_scores[metric.name] = computed_metrics
      else:
        multitask_scores[metric.name] = metric.compute_metric(
            y, y_pred, w, per_task_metrics=False, n_classes=n_classes)

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores
