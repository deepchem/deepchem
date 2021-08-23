import logging
import importlib.util
from typing import Optional, Union, Dict, List, Any
import os
import tensorflow as tf
import torch
import numpy as np

from deepchem.models import Model
from deepchem.models.logger import Logger

logs = logging.getLogger(__name__)
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
               project: Optional[str] = "deepchem",
               mode: Optional[str] = "online",
               id: Optional[str] = None,
               resume: Optional[Union[bool, str]] = None,
               anonymous: Optional[str] = "never",
               save_run_history: bool = False,
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
      logs.warning(
          'Note: Model checkpoints will not be uploaded to W&B in offline mode. '
          'Please set `mode="online"` if you need to log your model.')

    self.save_run_history = save_run_history

    # set wandb init arguments
    self.wandb_init_params: Dict[str, Any] = dict(
        name=name,
        project=project,
        entity=entity,
        mode=mode,
        id=id,
        resume=resume,
        anonymous=anonymous)
    self.wandb_init_params.update(**kwargs)
    self.initialized = False

    # Location ids are used to differentiate callbacks and logging locations seen by the logger
    self.location_ids: List[str] = []

  def setup(self, model: Model, **kwargs):
    """Initializes a W&B run and create a run object.
    If a pre-existing run is already initialized, use that instead.

    Parameters
    ----------
    model: Model
      DeepChem model object
    """
    self.model = model
    if self._wandb.run is None:
      self.wandb_run = self._wandb.init(**self.wandb_init_params)
    else:
      self.wandb_run = self._wandb.run

    # Add model settings to wandb config
    if (self.wandb_run is not None) and (self.model is not None):
      logger_config = dict(
          loss=self.model._loss_fn.loss,
          batch_size=self.model.batch_size,
          model_dir=self.model.model_dir,
          optimizer=self.model.optimizer,
          tensorboard=self.model.tensorboard,
          log_frequency=self.model.log_frequency)
      logger_config.update(**kwargs)
      self.wandb_run.config.update(logger_config)

    self.initialized = True

  def finish(self):
    """Finishes and closes the W&B run.
    Save run history data as field if configured to do that.
    """
    if self.save_run_history:
      history = self.wandb_run.history._data
      self.run_history = history
    if self.wandb_run is not None:
      self.wandb_run.finish()

  def log_batch(self,
                loss: Dict,
                step: int,
                inputs: tensor,
                labels: tensor,
                location: Optional[str] = None):
    """Log values for a single training batch.

    Parameters
    ----------
    loss: Dict
      the loss values for the batch
    step: int
      the current training step
    inputs: tensor
      batch input tensor
    labels: tensor
      batch labels tensor
    location: str, optional (default None)
      W&B chart panel section to log under
    """
    data = loss
    if location is not None:
      if location in self.location_ids:
        for key in list(data.keys()):
          new_key = location + "/" + str(key)
          data[new_key] = data.pop(key)
      else:
        self.location_ids.append(location)
        for key in list(data.keys()):
          new_key = location + "/" + str(key)
          data[new_key] = data.pop(key)

    # Log to W&B
    if self.wandb_run is not None:
      self.wandb_run.log(data, step=step)

  def log_values(self, data: Dict, step: int, location: Optional[str] = None):
    """Log values for a certain step in training/evaluation.

    Parameters
    ----------
    data: Dict
      data values to be logged
    step: int
      epoch number
    location: str, optional (default None)
      W&B chart panel section to log under
    """
    # Rename keys to the correct category
    if location is not None:
      if location in self.location_ids:
        for key in list(data.keys()):
          new_key = location + "/" + str(key)
          data[new_key] = data.pop(key)
      else:
        self.location_ids.append(location)
        for key in list(data.keys()):
          new_key = location + "/" + str(key)
          data[new_key] = data.pop(key)

    # Log to W&B
    if self.wandb_run is not None:
      self.wandb_run.log(data, step=step)

  def on_fit_end(self, data: Dict):
    """Called before the end of training.

    Parameters
    ----------
    data: Dict
      Training summary values to be logged
    """
    # Set summary
    if self.wandb_run is not None:
      if "global_step" in data:
        self.wandb_run.summary["global_step"] = data["global_step"]
      if "final_avg_loss" in data:
        self.wandb_run.summary["final_avg_loss"] = data["final_avg_loss"]

  def save_checkpoint(self,
                      checkpoint_name: str,
                      metadata: Optional[Dict] = None):
    """Save model checkpoint.

    Parameters
    ----------
    checkpoint_name: str
      name of the checkpoint
    metadata: Dict, optional(default None)
      metadata to be save along with the checkpoint
    """
    if (self.initialized is False) or (self.wandb_run is None):
      logs.warning(
          'WARNING: The wandb run has not been initialized. Please start training and in order to start checkpointing.'
      )
    else:
      if self.wandb_run.name is not None:
        model_name = checkpoint_name + "_" + self.wandb_run.name
      else:
        model_name = checkpoint_name
      artifact = self._wandb.Artifact(
          model_name, type='model', metadata=metadata)

      # Different saving mechanisms for different types of models
      if isinstance(self.model.model, tf.keras.Model):
        model_path = os.path.abspath(
            os.path.join(self.model.model_dir, model_name))
        self.model.model.save(model_path)
        artifact.add_dir(model_path)
      elif isinstance(self.model.model, torch.nn.Module):
        data = {
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.model._pytorch_optimizer.state_dict(),
            'global_step': self.model._global_step
        }

        saved_name = model_name + ".pt"
        model_path = os.path.abspath(
            os.path.join(self.model.model_dir, saved_name))
        torch.save(data, model_path)
        artifact.add_file(model_path)

      # apply aliases and log artifact
      step = self.model._global_step
      if not isinstance(step, int):
        # If tensorflow tensor convert to number
        step = step.numpy()

      aliases = ["latest", "step=" + str(step)]
      aliases = list(set(aliases))  # remove duplicates

      self.wandb_run.log_artifact(artifact, aliases=aliases)
