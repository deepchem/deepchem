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
               project: Optional[str] = None,
               save_dir: Optional[str] = None,
               mode: Optional[str] = "online",
               id: Optional[str] = None,
               resume: Optional[Union[bool, str]] = None,
               anonymous: Optional[str] = "never",
               save_run_history: bool = False,
               checkpoint_interval: int = 0,
               max_checkpoints_to_track: int = 1,
               model_dir: Optional[str] = None,
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
      logs.warning(
          'Note: Model checkpoints will not be uploaded to W&B in offline mode. '
          'Please set `mode="online"` if you need to log your model.')

    if checkpoint_interval == 0:
      logs.warning(
          'Note: WandbLogger model checkpointing is disabled since `checkpoint_interval = 0`. '
          'To enable checkpointing, please set `checkpoint_interval` to a positive integer value.'
      )

    if (checkpoint_interval is not None) and (checkpoint_interval >
                                              0) and (model_dir is None):
      raise ValueError(
          'Model checkpointing is active, but `model_dir` is not set. '
          'Please set `model_dir` to create a local location for your checkpoints.'
      )

    self.checkpoint_interval = checkpoint_interval
    self.max_checkpoints_to_track = max_checkpoints_to_track
    self.model_dir = model_dir

    self.save_dir = save_dir
    self.save_run_history = save_run_history

    # set wandb init arguments
    self.wandb_init_params: Dict[str, Any] = dict(
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

    # Location ids are used to differentiate callbacks and logging locations seen by the logger
    self.location_ids: List[str] = []

    # Keep track of best models during training and callbacks
    self.best_models: Dict = {}

  def setup(self, config: Dict):
    """Initializes a W&B run and create a run object.
    If a pre-existing run is already initialized, use that instead.
    """

    if self._wandb.run is None:
      self.wandb_run = self._wandb.init(**self.wandb_init_params)
      if self.wandb_run is not None:
        self.wandb_run.config.update(config)
    else:
      self.wandb_run = self._wandb.run
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
                location: Optional[str] = None,
                model: Optional[Model] = None,
                checkpoint_metric: Optional[str] = None,
                checkpoint_metric_value: Optional[numeric] = None,
                checkpoint_on_min: bool = True):
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

    # Save checkpoint
    if (checkpoint_metric is not None) and (checkpoint_metric_value is not None) and \
            (self.max_checkpoints_to_track is not None) and \
            (checkpoint_on_min is not None) and \
            (self.checkpoint_interval is not None) and \
            (self.model_dir is not None) and \
            (model is not None):
      if (self.checkpoint_interval >
          0) and (step % self.checkpoint_interval == 0):
        if location is None:
          location = "train"
        checkpoint_name = location.replace(
            "/", ".") + "." + checkpoint_metric + ".checkpoints"
        self._save_checkpoint(self.model_dir, model, checkpoint_name,
                              checkpoint_metric, checkpoint_metric_value,
                              self.max_checkpoints_to_track, checkpoint_on_min)

  def log_values(self,
                 data: Dict,
                 step: int,
                 location: Optional[str] = None,
                 model: Optional[Model] = None,
                 checkpoint_metric: Optional[str] = None,
                 checkpoint_metric_value: Optional[numeric] = None,
                 checkpoint_on_min: bool = True):
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

    # Save checkpoint
    if (checkpoint_metric is not None) and (checkpoint_metric_value is not None) and \
            (self.max_checkpoints_to_track is not None) and \
            (checkpoint_on_min is not None) and \
            (self.checkpoint_interval is not None) and \
            (self.model_dir is not None) and \
            (model is not None):
      if (self.checkpoint_interval > 0 is
          not None) and (self.checkpoint_interval >
                         0) and step % self.checkpoint_interval == 0:
        if location is None:
          location = "eval"
        checkpoint_name = location.replace(
            "/", ".") + "." + checkpoint_metric + ".checkpoints"
        self._save_checkpoint(self.model_dir, model, checkpoint_name,
                              checkpoint_metric, checkpoint_metric_value,
                              self.max_checkpoints_to_track, checkpoint_on_min)

  def on_fit_end(self,
                 data: Dict,
                 location: Optional[str] = None,
                 model: Optional[Model] = None,
                 checkpoint_metric: Optional[str] = None,
                 checkpoint_metric_value: Optional[numeric] = None,
                 checkpoint_on_min: Optional[bool] = True):
    # Set summary
    if self.wandb_run is not None:
      if "global_step" in data:
        self.wandb_run.summary["global_step"] = data["global_step"]
      if "final_avg_loss" in data:
        self.wandb_run.summary["final_avg_loss"] = data["final_avg_loss"]

    # Save checkpoint
    if (checkpoint_metric is not None) and \
            (checkpoint_metric_value is not None) and \
            (self.max_checkpoints_to_track is not None) and \
            (checkpoint_on_min is not None) and \
            (self.checkpoint_interval is not None) and \
            (self.model_dir is not None) and \
            (model is not None):
      if location is None:
        location = "train"
      checkpoint_name = location.replace(
          "/", ".") + "." + checkpoint_metric + ".checkpoints"
      self._save_checkpoint(self.model_dir, model, checkpoint_name,
                            checkpoint_metric, checkpoint_metric_value,
                            self.max_checkpoints_to_track, checkpoint_on_min)

  def _save_checkpoint(self,
                       path: str,
                       dc_model: Model,
                       checkpoint_name: str,
                       value_name: str,
                       value: numeric,
                       max_checkpoints_to_track: int,
                       checkpoint_on_min: bool,
                       metadata: Optional[Dict] = None):

    # Only called once when first checkpoint is saved to create tracking record
    if (checkpoint_name not in self.best_models):
      # Set up to track top values of this checkpoint
      self.best_models[checkpoint_name] = {}
      self.best_models[checkpoint_name]["model_values"] = []

    # Sort in order
    if checkpoint_on_min:
      self.best_models[checkpoint_name]["model_values"] = sorted(
          self.best_models[checkpoint_name][
              "model_values"])[:max_checkpoints_to_track]
    else:
      self.best_models[checkpoint_name]["model_values"] = sorted(
          self.best_models[checkpoint_name]["model_values"],
          reverse=True)[:max_checkpoints_to_track]

    # Save checkpoint only if it passes the cutoff in the model values
    should_save = False
    if len(self.best_models[checkpoint_name]
           ["model_values"]) < max_checkpoints_to_track:
      # first checkpoint to be saved
      should_save = True
    elif checkpoint_on_min and (
        value < self.best_models[checkpoint_name]["model_values"][-1]):
      # value passes minimum cut off
      should_save = True
    elif (not checkpoint_on_min) and (
        value > self.best_models[checkpoint_name]["model_values"][-1]):
      # value passes maximum cut off
      should_save = True

    if (self.initialized is False) or (self.wandb_run is None):
      logs.warning(
          'WARNING: The wandb run has not been initialized. Please start training and in order to start checkpointing.'
      )
    else:
      if should_save:
        self.best_models[checkpoint_name]["model_values"].append(value)
        if self.wandb_run.name is not None:
          model_name = checkpoint_name + "_" + self.wandb_run.name
        else:
          model_name = checkpoint_name
        artifact = self._wandb.Artifact(
            model_name, type='model', metadata=metadata)

        # Different saving mechanisms for different types of models
        if isinstance(dc_model.model, tf.keras.Model):
          model_path = os.path.abspath(os.path.join(path, model_name))
          dc_model.model.save(model_path)
          artifact.add_dir(model_path)

        elif isinstance(dc_model.model, torch.nn.Module):
          data = {
              'model_state_dict': dc_model.model.state_dict(),
              'optimizer_state_dict': dc_model._pytorch_optimizer.state_dict(),
              'global_step': dc_model._global_step
          }

          saved_name = model_name + ".pt"
          model_path = os.path.abspath(os.path.join(path, saved_name))
          torch.save(data, model_path)
          artifact.add_file(model_path)

        # apply aliases and log artifact
        step = dc_model._global_step
        if not isinstance(step, int):
          # If tensorflow tensor convert to number
          step = step.numpy()
        aliases = ["latest", "step=" + str(step), value_name + "=" + str(value)]
        aliases = list(set(aliases))  # remove duplicates

        self.wandb_run.log_artifact(artifact, aliases=aliases)
