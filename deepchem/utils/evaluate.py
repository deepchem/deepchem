"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

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
    if mode == "classification":
      y_pred = self.model.predict_proba(self.dataset, self.output_transformers)
      y_pred_print = self.model.predict(self.dataset,
                                        self.output_transformers).astype(int)
    else:
      y_pred = self.model.predict(self.dataset, self.output_transformers)
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

  def __init__(self,
               model,
               generator,
               transformers,
               labels,
               outputs=None,
               n_tasks=1,
               n_classes=2,
               weights=list()):
    """
    Parameters
    ----------
    model: Model
      Model to evaluate
    generator: Generator
      Generator which yields {layer: numpyArray} to feed into model
    transformers:
      Tranformers to "undo" when applied to the models outputs
    labels: list of Layer
      layers which are keys in the generator to compare to outputs
    outputs: list of Layer
      if None will use the outputs of the model
    weights: np.array
      Must be of the shape (n_samples, n_tasks)
      if weights[sample][task] is 0 that sample will not be used
      for computing the task metric
    """
    self.model = model
    self.generator = generator
    self.n_tasks = n_tasks
    self.n_classes = n_classes
    self.output_transformers = [
        transformer for transformer in transformers if transformer.transform_y
    ]
    if outputs is None:
      self.output_keys = model.outputs
    else:
      self.output_keys = outputs
    self.label_keys = labels
    self.weights = weights
    if len(self.label_keys) != len(self.output_keys):
      raise ValueError("Must have same number of labels and outputs")

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
    self.model.build()
    y = []
    w = []

    def generator_closure():
      for feed_dict in self.generator:
        labels = []
        for layer in self.label_keys:
          labels.append(feed_dict[layer])
          del feed_dict[layer]
        for weight in self.weights:
          w.append(feed_dict[weight])
          del feed_dict[weight]
        y.append(np.array(labels))
        yield feed_dict

    if not len(metrics):
      return {}
    else:
      mode = metrics[0].mode
    if mode == "classification":
      y_pred = self.model.predict_proba_on_generator(generator_closure())
      y = np.transpose(np.array(y), axes=[0, 2, 1, 3])
      y = np.reshape(y, newshape=(-1, self.n_tasks, self.n_classes))
      y = from_one_hot(y, axis=-1)
    else:
      y_pred = self.model.predict_proba_on_generator(generator_closure())
      y = np.transpose(np.array(y), axes=[0, 2, 1, 3])
      y = np.reshape(y, newshape=(-1, self.n_tasks))
      y_pred = np.reshape(y_pred, newshape=(-1, self.n_tasks))
    multitask_scores = {}
    all_task_scores = {}

    y = undo_transforms(y, self.output_transformers)
    y_pred = undo_transforms(y_pred, self.output_transformers)
    if len(w) != 0:
      w = np.array(w)
      w = np.reshape(w, newshape=y.shape)

    # Compute multitask metrics
    for metric in metrics:
      if per_task_metrics:
        multitask_scores[metric.name], computed_metrics = metric.compute_metric(
            y, y_pred, w, per_task_metrics=True, n_classes=self.n_classes)
        all_task_scores[metric.name] = computed_metrics
      else:
        multitask_scores[metric.name] = metric.compute_metric(
            y, y_pred, w, per_task_metrics=False, n_classes=self.n_classes)

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores
