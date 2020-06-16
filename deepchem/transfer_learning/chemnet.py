"""
General purpose transfer learning script. Currently supports only mode transfer
(classification to regression and vice-versa) and different multi-tasking options.
For more advanced finetuning, use the load_from_pretrained function as part of
deepchem.models.KerasModel
"""

__author__ = "Vignesh Ram Somnath"
__license_ = "MIT"

import numpy as np
import tensorflow as tf
import os
import logging
import deepchem
import tempfile

from deepchem.models.tensorgraph.optimizers import RMSProp
from deepchem.molnet.load_function.chembl25_datasets import chembl25_tasks
from deepchem.models.callbacks import ValidationCallback, EarlyStoppingCallBack
from deepchem.transfer_learning import get_model_str

logger = logging.getLogger(__name__)

BASE_URL = "https://s3-us-west-1.amazonaws.com/deepchem.io/checkpoints/"


def pretrain_model(chembl_config,
                   model_class,
                   model_hparams,
                   train_config,
                   metric_fn=None,
                   **kwargs):
  """Pretraining the model on ChEMBL dataset.

    Parameters
    ----------
    chembl_config: dict,
        Arguments used for loading ChEMBL dataset
    model_class: dc.models.KerasModel
        Model class
    model_hparams: dict,
        Dictionary of model arguments
    train_config: dict,
        Dictionary of training options
    metric_fn: deepchem.metrics.Metric, default None
        Metric used for evaluating training progress
    """
  if metric_fn is None:
    metric_fn = deepchem.metrics.rms_score
  metric = deepchem.metrics.Metric(
      metric_fn, task_averager=np.mean, mode="regression", verbose=False)

  tasks, dataset, transformers = deepchem.molnet.load_chembl25(**chembl_config)
  train, valid, test = dataset

  model_hparams = update_hparams(model_hparams)
  if model_hparams.get('model_dir', None) is None:
    model_dir = tempfile.mkdtemp()
  else:
    model_dir = model_hparams.get('model_dir')

  n_samples = len(train.w)
  batch_size = model_hparams.get('batch_size')
  nb_epoch = train_config.get('nb_epoch', 50)
  patience_early_stopping = train_config.get('patience_early_stopping', 10)

  if n_samples % batch_size == 0:
    steps_per_epoch = n_samples // batch_size
  else:
    steps_per_epoch = 1 + (n_samples // batch_size)

  callbacks = [
      ValidationCallback(
          dataset=valid, metrics=[metric], interval=steps_per_epoch),
      EarlyStoppingCallBack(
          dataset=valid,
          metric='loss',
          interval=steps_per_epoch,
          patience=patience_early_stopping,
          save_dir=model_dir)
  ]

  model = model_class(n_tasks=len(tasks), **model_hparams)
  model.fit(train, nb_epoch, checkpoint_interval=0, callbacks=callbacks)
  test_scores = model.evaluate(
      test, metrics=[metric], transformers=transformers)
  logger.info("Test-{}: {}".format(metric.name, test_scores[metric.name]))
  return model


def finetune_model(train_dataset,
                   valid_dataset,
                   test_dataset,
                   transformers,
                   tasks,
                   model_class,
                   model_hparams,
                   train_config,
                   pretrained_model=None,
                   pretrain_config=None,
                   metric_type=None,
                   include_top=False,
                   model_dir=None,
                   **kwargs):
  if model_hparams.get('mode', None) is None:
    raise ValueError(
        "Finetuning task must have a classification or regression mode defined."
    )

  if metric_type is None:
    if model_hparams['mode'] == "classification":
      metric_type = deepchem.metrics.roc_auc_score
    else:
      metric_type = deepchem.metrics.rms_score

  if model_hparams.get('model_dir', None) is None:
    model_dir = tempfile.mkdtemp()
  else:
    model_dir = model_hparams.get('model_dir')

  if pretrained_model is None:
    if pretrain_config is None:
      raise ValueError(
          "Pretrain config cannot be None when pretrained model is None.")

    model_str = get_model_str(model_class.__name__, model_hparams)
    ckpt_url = BASE_URL + model_str + "weights.zip"
    deepchem.utils.download_url(url=ckpt_url, dest_dir=model_dir)
    deepchem.utils.unzip_file(
        file=os.path.join(model_dir, "weights.zip"), dest_dir=model_dir)

    pretrained_model = model_class(
        **pretrain_config, n_tasks=len(chembl25_tasks), mode="regression")

  if not model_hparams.get('mode', None):
    raise ValueError(
        "Task mode of the model cannot be None. Should be one of classification or regression."
    )

  metric = deepchem.metrics.Metric(
      metric_type,
      task_averager=np.mean,
      mode=model_hparams["mode"],
      verbose=False)

  model_hparams = update_hparams(model_hparams)
  if not model_hparams.get('model_dir'):
    model_hparams['model_dir'] = model_dir
  model = model_class(n_tasks=len(tasks), **model_hparams)

  nb_epoch = train_config.get('nb_epoch', 500)
  patience = train_config.get('patience', 50)
  delta = train_config.get('delta', 0.01)

  model.load_from_pretrained(
      source_model=pretrained_model,
      assignment_map=None,
      value_map=None,
      checkpoint=None,
      model_dir=model_dir,
      include_top=include_top)

  callbacks = [
      ValidationCallback(
          dataset=valid_dataset,
          metrics=[metric],
          interval=steps_per_epoch,
          early_stop_metric='loss',
          interval=steps_per_epoch,
          patience=patience,
          save_dir=model_dir,
          delta=delta)
  ]

  model.fit(train_dataset, nb_epoch, checkpoint_interval=0, callbacks=callbacks)
  test_scores = model.evaluate(
      test_dataset, metrics=[metric], transformers=transformers)
  logger.info("Test-{}: {}".format(metric.name, test_scores[metric.name]))
  return model


def update_hparams(hparams):
  """Adds default generic KerasModel arguments if missing.

    Parameters
    ----------
    hparams: dict,
      Dictionary of model hyperparameters
    """
  if not hparams.get('batch_size'):
    hparams['batch_size'] = 32
  if not hparams.get('learning_rate'):
    hparams['learning_rate'] = 1e-3
  if not hparams.get('optimizer'):
    hparams['optimizer'] = RMSProp(learning_rate=hparams['learning_rate'])
  if tf.executing_eagerly() or (not hparams.get('tensorboard')):
    hparams['tensorboard'] = False
  if hparams['tensorboard'] and (not hparams.get('tensorboard_log_frequency')):
    hparams['tensorboard_log_frequency'] = 100

  return hparams
