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
  test_scores = model.evaluate(test, metrics=[metric], [])
  logger.info("Test-{}: {}".format(metric.name, test_scores[metric.name]))
  return model


def finetune_model(dataset_name,
                   dataset_config,
                   model_class,
                   model_hparams,
                   train_config,
                   pretrained_model=None,
                   pretrain_config=None,
                   metric_type=None,
                   include_top=False,
                   model_dir=None,
                   **kwargs):
  if metric_type is None:
    if model_hparams['mode'] == "classification":
      metric_type = deepchem.metrics.roc_auc_score
    else:
      metric_type = deepchem.metrics.rms_score

  loading_functions = {
      'bace_c': deepchem.molnet.load_bace_classification,
      'bace_r': deepchem.molnet.load_bace_regression,
      'bbbp': deepchem.molnet.load_bbbp,
      'chembl': deepchem.molnet.load_chembl,
      'clearance': deepchem.molnet.load_clearance,
      'clintox': deepchem.molnet.load_clintox,
      'delaney': deepchem.molnet.load_delaney,
      'factors': deepchem.molnet.load_factors,
      'hiv': deepchem.molnet.load_hiv,
      'hopv': deepchem.molnet.load_hopv,
      'hppb': deepchem.molnet.load_hppb,
      'kaggle': deepchem.molnet.load_kaggle,
      'kinase': deepchem.molnet.load_kinase,
      'lipo': deepchem.molnet.load_lipo,
      'muv': deepchem.molnet.load_muv,
      'nci': deepchem.molnet.load_nci,
      'pcba': deepchem.molnet.load_pcba,
      'pcba_146': deepchem.molnet.load_pcba_146,
      'pcba_2475': deepchem.molnet.load_pcba_2475,
      'pdbbind': deepchem.molnet.load_pdbbind_grid,
      'ppb': deepchem.molnet.load_ppb,
      'qm7': deepchem.molnet.load_qm7_from_mat,
      'qm7b': deepchem.molnet.load_qm7b_from_mat,
      'qm8': deepchem.molnet.load_qm8,
      'qm9': deepchem.molnet.load_qm9,
      'sampl': deepchem.molnet.load_sampl,
      'sider': deepchem.molnet.load_sider,
      'sweetlead': deepchem.molnet.load_sweet,
      'thermosol': deepchem.molnet.load_thermosol,
      'tox21': deepchem.molnet.load_tox21,
      'toxcast': deepchem.molnet.load_toxcast,
      'uv': deepchem.molnet.load_uv,
  }

  if pretrained_model is None:
    if pretrain_config is None:
      raise ValueError(
          "Pretrain config cannot be None when pretrained model is None.")

    model_str = get_model_str(model_class.__name__, model_hparams)
    pretrained_model = model_class(
        **pretrain_config, n_tasks=len(chembl25_tasks), mode="regression")

  load_fn = loading_functions[dataset_name]
  tasks, dataset, transformers = load_fn(**dataset_config)

  if not model_hparams.get('mode', None):
    raise ValueError(
        "Task mode of the model cannot be None. Should be one of classification or regression."
    )

  metric = deepchem.metrics.Metric(
      metric_type,
      task_averager=np.mean,
      mode=model_hparams["mode"],
      verbose=False)
  train, valid, test = dataset

  model_hparams = update_hparams(model_hparams)
  if not model_hparams.get('model_dir'):
    model_hparams['model_dir'] = model_dir
  model = model_class(n_tasks=len(tasks), **model_hparams)

  nb_epoch = train_config.get('nb_epoch', 500)
  patience_early_stopping = train_config.get('patience_early_stopping', 50)

  model.load_from_pretrained(
      source_model=pretrained_model,
      assignment_map=None,
      value_map=None,
      checkpoint=None,
      model_dir=model_dir,
      include_top=include_top)

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

  model.fit(train, nb_epoch, checkpoint_interval=0, callbacks=callbacks)
  test_scores = model.evaluate(test, metrics=[metric], [])
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
