"""
Utility functions to evaluate models on datasets.
"""
import csv
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import deepchem as dc
from deepchem.metrics import Metric

logger = logging.getLogger(__name__)

Score = Dict[str, float]
Metric_Func = Callable[..., Any]
Metrics = Union[Metric, Metric_Func, List[Metric], List[Metric_Func]]


def output_statistics(scores: Score, stats_out: str) -> None:
  """Write computed stats to file.

  Statistics are written to specified `stats_out` file.

  Parameters
  ----------
  scores: dict
    Dictionary mapping names of metrics to scores.
  stats_out: str
    Name of file to write scores to.
  """
  logger.warning("output_statistics is deprecated.")
  with open(stats_out, "w") as statsfile:
    statsfile.write(str(scores) + "\n")


def output_predictions(dataset: "dc.data.Dataset", y_preds: np.ndarray,
                       csv_out: str) -> None:
  """Writes predictions to file.

  Writes predictions made on `dataset` to a specified file on
  disk. `dataset.ids` are used to format predictions. The produce CSV file will have format as follows

  | ID          | Task1Name    | Task2Name    |
  | ----------- | ------------ | ------------ |
  | identifer1  | prediction11 | prediction12 |
  | identifer2  | prediction21 | prediction22 |

  Parameters
  ----------
  dataset: dc.data.Dataset
    Dataset on which predictions have been made.
  y_preds: np.ndarray
    Predictions to output
  csv_out: str
    Name of file to write predictions to.
  """
  data_ids = dataset.ids
  n_tasks = len(dataset.get_task_names())
  y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
  assert len(y_preds) == len(data_ids)
  with open(csv_out, "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["ID"] + dataset.get_task_names())
    for mol_id, y_pred in zip(data_ids, y_preds):
      csvwriter.writerow([mol_id] + list(y_pred))


def _process_metric_input(metrics: Metrics) -> List[Metric]:
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
  if not isinstance(metrics, list):
    # FIXME: Incompatible types in assignment
    metrics = [metrics]  # type: ignore

  final_metrics = []
  # FIXME: Argument 1 to "enumerate" has incompatible type
  for i, metric in enumerate(metrics):  # type: ignore
    # Ensure that metric is wrapped in a list.
    if isinstance(metric, Metric):
      final_metrics.append(metric)
    # This case checks if input is a function then wraps a
    # dc.metrics.Metric object around it
    elif callable(metric):
      wrap_metric = Metric(metric, name="metric-%d" % (i + 1))
      final_metrics.append(wrap_metric)
    else:
      raise ValueError(
          "metrics must be one of metric function / dc.metrics.Metric object /"
          "list of dc.metrics.Metric or metric functions.")
  return final_metrics


def relative_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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


class Evaluator(object):
  """Class that evaluates a model on a given dataset.

  The evaluator class is used to evaluate a `dc.models.Model` class on
  a given `dc.data.Dataset` object. The evaluator is aware of
  `dc.trans.Transformer` objects so will automatically undo any
  transformations which have been applied.

  Examples
  --------
  Evaluators allow for a model to be evaluated directly on a Metric
  for `sklearn`. Let's do a bit of setup constructing our dataset and
  model.

  >>> import deepchem as dc
  >>> import numpy as np
  >>> X = np.random.rand(10, 5)
  >>> y = np.random.rand(10, 1)
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>> model = dc.models.MultitaskRegressor(1, 5)
  >>> transformers = []

  Then you can evaluate this model as follows
  >>> import sklearn
  >>> evaluator = Evaluator(model, dataset, transformers)
  >>> multitask_scores = evaluator.compute_model_performance(
  ...     sklearn.metrics.mean_absolute_error)

  Evaluators can also be used with `dc.metrics.Metric` objects as well
  in case you want to customize your metric further.

  >>> evaluator = Evaluator(model, dataset, transformers)
  >>> metric = dc.metrics.Metric(dc.metrics.mae_score)
  >>> multitask_scores = evaluator.compute_model_performance(metric)
  """

  def __init__(self, model, dataset: "dc.data.Dataset",
               transformers: List["dc.trans.Transformer"]):
    """Initialize this evaluator

    Parameters
    ----------
    model: Model
      Model to evaluate. Note that this must be a regression or
      classification model and not a generative model.
    dataset: Dataset
      Dataset object to evaluate `model` on.
    transformers: List[Transformer]
      List of `dc.trans.Transformer` objects. These transformations
      must have been applied to `dataset` previously. The dataset will
      be untransformed for metric evaluation.
    """

    self.model = model
    self.dataset = dataset
    self.output_transformers = [
        transformer for transformer in transformers if transformer.transform_y
    ]

  def output_statistics(self, scores: Score, stats_out: str):
    """ Write computed stats to file.

    Parameters
    ----------
    scores: dict
      Dictionary mapping names of metrics to scores.
    stats_out: str
      Name of file to write scores to.
    """
    logger.warning(
        "Evaluator.output_statistics is deprecated."
        "Please use dc.utils.evaluate.output_statistics instead."
        "This method will be removed in a future version of DeepChem.")
    with open(stats_out, "w") as statsfile:
      statsfile.write(str(scores) + "\n")

  def output_predictions(self, y_preds: np.ndarray, csv_out: str):
    """Writes predictions to file.

    Writes predictions made on `self.dataset` to a specified file on
    disk. `self.dataset.ids` are used to format predictions.

    Parameters
    ----------
    y_preds: np.ndarray
      Predictions to output
    csv_out: str
      Name of file to write predictions to.
    """
    logger.warning(
        "Evaluator.output_predictions is deprecated."
        "Please use dc.utils.evaluate.output_predictions instead."
        "This method will be removed in a future version of DeepChem.")
    data_ids = self.dataset.ids
    n_tasks = len(self.dataset.get_task_names())
    y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
    assert len(y_preds) == len(data_ids)
    with open(csv_out, "w") as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(["ID"] + self.dataset.get_task_names())
      for mol_id, y_pred in zip(data_ids, y_preds):
        csvwriter.writerow([mol_id] + list(y_pred))

  def compute_model_performance(
      self,
      metrics: Metrics,
      csv_out: Optional[str] = None,
      stats_out: Optional[str] = None,
      per_task_metrics: bool = False,
      use_sample_weights: bool = False,
      n_classes: int = 2) -> Union[Score, Tuple[Score, Score]]:
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
      `np.ndarray` objects and return a floating point score. The
      metric function may also accept a keyword argument
      `sample_weight` to account for per-sample weights.
    csv_out: str, optional (DEPRECATED)
      Filename to write CSV of model predictions.
    stats_out: str, optional (DEPRECATED)
      Filename to write computed statistics.
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.
    use_sample_weights: bool, optional (default False)
      If set, use per-sample weights `w`.
    n_classes: int, optional (default None)
      If specified, will use `n_classes` as the number of unique classes
      in `self.dataset`. Note that this argument will be ignored for
      regression metrics.

    Returns
    -------
    multitask_scores: dict
      Dictionary mapping names of metrics to metric scores.
    all_task_scores: dict, optional
      If `per_task_metrics == True`, then returns a second dictionary
      of scores for each task separately.
    """
    if csv_out is not None:
      logger.warning(
          "csv_out is deprecated as an argument and will be removed in a future version of DeepChem."
          "Output is not written to CSV; manually write output instead.")
    if stats_out is not None:
      logger.warning(
          "stats_out is deprecated as an argument and will be removed in a future version of DeepChem."
          "Stats output is not written; please manually write output instead")
    # Process input metrics
    metrics = _process_metric_input(metrics)

    y = self.dataset.y
    y = dc.trans.undo_transforms(y, self.output_transformers)
    w = self.dataset.w

    y_pred = self.model.predict(self.dataset, self.output_transformers)
    n_tasks = len(self.dataset.get_task_names())

    multitask_scores = {}
    all_task_scores = {}

    # Compute multitask metrics
    for metric in metrics:
      results = metric.compute_metric(
          y,
          y_pred,
          w,
          per_task_metrics=per_task_metrics,
          n_tasks=n_tasks,
          n_classes=n_classes,
          use_sample_weights=use_sample_weights)
      if per_task_metrics:
        multitask_scores[metric.name], computed_metrics = results
        all_task_scores[metric.name] = computed_metrics
      else:
        multitask_scores[metric.name] = results

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores


class GeneratorEvaluator(object):
  """Evaluate models on a stream of data.

  This class is a partner class to `Evaluator`. Instead of operating
  over datasets this class operates over a generator which yields
  batches of data to feed into provided model.

  Examples
  --------
  >>> import deepchem as dc
  >>> import numpy as np
  >>> X = np.random.rand(10, 5)
  >>> y = np.random.rand(10, 1)
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>> model = dc.models.MultitaskRegressor(1, 5)
  >>> generator = model.default_generator(dataset, pad_batches=False)
  >>> transformers = []

  Then you can evaluate this model as follows

  >>> import sklearn
  >>> evaluator = GeneratorEvaluator(model, generator, transformers)
  >>> multitask_scores = evaluator.compute_model_performance(
  ...     sklearn.metrics.mean_absolute_error)

  Evaluators can also be used with `dc.metrics.Metric` objects as well
  in case you want to customize your metric further. (Note that a given
  generator can only be used once so we have to redefine the generator here.)

  >>> generator = model.default_generator(dataset, pad_batches=False)
  >>> evaluator = GeneratorEvaluator(model, generator, transformers)
  >>> metric = dc.metrics.Metric(dc.metrics.mae_score)
  >>> multitask_scores = evaluator.compute_model_performance(metric)
  """

  def __init__(self,
               model,
               generator: Iterable[Tuple[Any, Any, Any]],
               transformers: List["dc.trans.Transformer"],
               labels: Optional[List] = None,
               weights: Optional[List] = None):
    """
    Parameters
    ----------
    model: Model
      Model to evaluate.
    generator: generator
      Generator which yields batches to feed into the model. For a
      KerasModel, it should be a tuple of the form (inputs, labels,
      weights). The "correct" way to create this generator is to use
      `model.default_generator` as shown in the example above.
    transformers: List[Transformer]
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

  def compute_model_performance(
      self,
      metrics: Metrics,
      per_task_metrics: bool = False,
      use_sample_weights: bool = False,
      n_classes: int = 2) -> Union[Score, Tuple[Score, Score]]:
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
    use_sample_weights: bool, optional (default False)
      If set, use per-sample weights `w`.
    n_classes: int, optional (default None)
      If specified, will assume that all `metrics` are classification
      metrics and will use `n_classes` as the number of unique classes
      in `self.dataset`.

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
      """This function is used to pull true labels/weights out as we iterate over the generator."""
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
    y_true = dc.trans.undo_transforms(y, self.output_transformers)
    y_pred = dc.trans.undo_transforms(y_pred, self.output_transformers)

    # Compute multitask metrics
    for metric in metrics:
      results = metric.compute_metric(
          y_true,
          y_pred,
          w,
          per_task_metrics=per_task_metrics,
          n_classes=n_classes,
          use_sample_weights=use_sample_weights)
      if per_task_metrics:
        multitask_scores[metric.name], computed_metrics = results
        all_task_scores[metric.name] = computed_metrics
      else:
        multitask_scores[metric.name] = results

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores
