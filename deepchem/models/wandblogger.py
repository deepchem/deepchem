import logging
import importlib.util
from typing import Optional, Union, Dict, List
import shutil
import os
from distutils.dir_util import copy_tree
import tensorflow as tf
import torch
import numpy as np
from deepchem.models.logger import Logger

logger = logging.getLogger(__name__)
numeric = Union[tf.Tensor, torch.Tensor, int, float, complex, np.number]
tensor = Union[tf.Tensor, torch.Tensor]

def is_wandb_available():
  return importlib.util.find_spec("wandb") is not None


class WandbLogger(Logger):
  """Weights & Biases Logger.

    This is a logger class that can be passed into the initialization
    of a KerasModel or TorchModel. It initializes and sets up a wandb logger which
    will log the specified metrics calculated on the specific datasets
    to the user's W&B dashboard.

    If a WandbLogger is provided to the wandb_logger flag,
    the metrics are logged to Weights & Biases, along with other information
    such as epoch number, losses, sample counts, and model configuration data.
    """

  def __init__(self,
               name: Optional[str] = None,
               entity: Optional[str] = None,
               project: Optional[str] = None,
               save_dir: Optional[str] = None,
               mode: Optional[str] = "online",
               id: Optional[str] = None,
               resume: Optional[Union[bool, str]] = None,
               anonymous: Optional[str] = "never",
               save_run_history: Optional[bool] = False,
               **kwargs):
    """Creates a WandbLogger.

    Parameters
    ----------
    name: str
      a display name for the run in the W&B dashboard
    entity: str
      an entity is a username or team name where you're sending the W&B run
    project: str
      the name of the project where you're sending the new W&B run
    save_dir: str
      path where data is saved (wandb dir by default)
    mode: str
      W&B online or offline mode
    id: str
      a unique ID for this run, used for resuming
    resume: bool or str
      sets the resuming behavior
    anonymous: str
      controls anonymous data logging
    save_run_history: bool
      whether to save the run history to the logger at the end (for testing purposes)
    """

    assert is_wandb_available(
    ), "WandbLogger requires wandb to be installed. Please run `pip install wandb --upgrade`"
    import wandb
    self._wandb = wandb

    if mode == "offline":
      logger.warning(
          'Note: Model checkpoints will not be uploaded to W&B in offline mode.\n'
          'Please set `mode="online"` if you need to log your model.')

    self.save_dir = save_dir
    self.save_run_history = save_run_history

    # set wandb init arguments
    self.wandb_init_params = dict(
        name=name,
        project=project,
        entity=entity,
        mode=mode,
        id=id,
        dir=save_dir,
        resume=resume,
        anonymous=anonymous)
    self.wandb_init_params.update(**kwargs)
    self.initialized = False

    # Dataset ids are used to differentiate datasets seen by the logger
    self.dataset_ids: List[Union[int, str]] = []

  def setup(self, config):
    """Initializes a W&B run and create a run object.
    If a pre-existing run is already initialized, use that instead.
    """

    if self._wandb.run is None:
      self.wandb_run = self._wandb.init(**self.wandb_init_params)
      self.wandb_run.config.update(config)
    else:
      self.wandb_run = self._wandb.run
    self.initialized = True

  def log_values(self, values: Dict, step: int, group=None, dataset_id=None):
    data = values
    # Log into the correct category
    if group is not None:
      data = {group + '/' + k: v for k, v in values.items()}

    # Log unique keys for each dataset
    if dataset_id is not None:
      if dataset_id in self.dataset_ids:
        for key in list(data.keys()):
          idx = self.dataset_ids.index(dataset_id)
          new_key = str(key) + "_(" + str(idx) + ")"
          data[new_key] = data.pop(key)
      else:
        self.dataset_ids.append(dataset_id)
        for key in list(data.keys()):
          idx = self.dataset_ids.index(dataset_id)
          new_key = str(key) + "_(" + str(idx) + ")"
          data[new_key] = data.pop(key)

    self.wandb_run.log(data, step=step)

  def log_batch(self, loss: Dict, step: int, inputs: tensor, labels: tensor, group=None):
    data = loss
    # Log into the correct category (for example: "train", "eval")
    if group is not None:
      data = {group + '/' + k: v for k, v in loss.items()}

    self.wandb_run.log(data, step=step)

  def log_epoch(self, data: Dict, epoch: int):
    pass

  def finish(self):
    """Finishes and closes the W&B run.
    Save run history data as field if configured to do that.
    """
    if self.save_run_history:
      history = self.wandb_run.history._data
      self.run_history = history
    self.wandb_run.finish()

  def save_model(self, path):
    abs_path = os.path.abspath(path)
    abs_path = abs_path.replace("/", ".")
    artifact = self._wandb.Artifact(abs_path, type='model')
    artifact.add_dir(path)
    self.wandb_run.log_artifact(artifact)

    #
    # path_list = path.split(os.sep)
    # # destination folder will have same name as save directory
    # dest = os.path.join(self.wandb_run.dir, path_list[-1])
    # shutil.rmtree(dest, ignore_errors=True) # clear dest folder to avoid file already exist error
    # checkpoint_names = ["ckpt", "checkpoint", ".pt", ".pth"]
    # # Copy all checkpoint files to wandb.run.dir for upload when run finishes
    # for file in os.listdir(path):
    #     if any(substring in file.lower() for substring in checkpoint_names):
    #
    #         if not os.path.exists(dest):
    #             os.makedirs(dest)
    #
    #         shutil.copy2(os.path.join(path, file), os.path.join(dest, file))
