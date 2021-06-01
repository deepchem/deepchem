import math
import copy
import logging
import importlib.util
from typing import List, Optional, Union
from deepchem.data import Dataset
from deepchem.metrics import Metric
from deepchem.models.callbacks import ValidationCallback

logger = logging.getLogger(__name__)


def is_wandb_available():
  return importlib.util.find_spec("wandb") is not None


class WandbLogger(object):
  """Weights & Biases Logger for KerasModel.

    This is a logger class that can be passed into the initialization
    of a KerasModel. It initializes and sets up a wandb logger which
    will log the specified metrics calculated on the specific datasets
    to the user's W&B dashboard.

    If a WandbLogger is provided to the wandb_logger flag in KerasModel,
    the metrics are logged to Weights & Biases, along with other information
    such as epoch number, losses, sample counts, and model configuration data.
    """

  def __init__(self,
               train_dataset: Dataset,
               eval_dataset: Optional[Dataset] = None,
               metrics: Optional[List[Metric]] = None,
               logging_strategy: Optional[str] = "step",
               name: Optional[str] = None,
               entity: Optional[str] = None,
               project: Optional[str] = None,
               save_dir: Optional[str] = None,
               mode: Optional[str] = "online",
               id: Optional[str] = None,
               resume: Optional[Union[bool, str]] = None,
               anonymous: Optional[str] = "never",
               log_model: Optional[bool] = False,
               log_dataset: Optional[bool] = False,
               save_run_history: Optional[bool] = False,
               **kwargs):
    """Parameters
    ----------
    train_dataset: dc.data.Dataset
      the training set on which the model is run on
    eval_dataset: dc.data.Dataset
      the validation set on which to compute the metrics
    metrics: list of dc.metrics.Metric
      metrics to compute on eval_dataset
    logging_strategy: str
      the logging strategy used for logging (step or epoch)
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
    log_model: bool
      whether to log the model to W&B
    log_dataset: bool
      whether to log the dataset to W&B
    save_run_history: bool
      whether to save the run history to the logger at the end (for testing purposes)
    """

    assert is_wandb_available(
    ), "WandbLogger requires wandb to be installed. Please run `pip install wandb --upgrade`"
    import wandb
    self._wandb = wandb

    if mode == "offline" and log_model:
      raise Exception(
          f'Providing log_model={log_model} and mode={mode} is an invalid configuration'
          ' since model checkpoints cannot be uploaded in offline mode.\n'
          'Hint: Set `mode="online"` to log your model.')

    # Check for metrics and logging strategy
    if ((metrics is None) or (not metrics)) and (eval_dataset is not None):
      logger.warning(
          "Warning: No metrics are provided. "
          "Please provide a list of metrics to be calculated on the datasets.")

    if logging_strategy != "step" and logging_strategy != "epoch":
      logger.warning(
          "Warning: `logging_strategy` needs to be either 'step' or 'epoch'. Defaulting to 'step'."
      )
      logging_strategy = "step"

    self.datasets = {"train": train_dataset, "eval": eval_dataset}
    self.train_dataset_size = len(self.datasets["train"])
    self.metrics = metrics
    self.logging_strategy = logging_strategy

    self.log_model = log_model
    self.log_dataset = log_dataset
    self.save_dir = save_dir
    self.save_run_history = save_run_history

    # set wandb init arguments
    self.wandb_init_params = dict(name=name,
                                  project=project,
                                  entity=entity,
                                  mode=mode,
                                  id=id,
                                  dir=save_dir,
                                  resume=resume,
                                  anonymous=anonymous)
    self.wandb_init_params.update(**kwargs)
    self.initialized = False

  def setup(self):
    """Initializes a W&B run and create a run object.
    """
    self.wandb_run = self._wandb.init(**self.wandb_init_params)
    self.initialized = True

  def check_other_loggers(self, callbacks):
    """Check for different callbacks and warn for redundant logging behaviour.
    Parameters
    ----------
    callbacks: function or list of functions
      one or more functions of the form f(model, step) that will be passed into fit().

    """
    for c in callbacks:
      if isinstance(c, ValidationCallback):
        logger.warning(
            "Note: You are using both WandbLogger and ValidationCallback. "
            "This will result in evaluation metrics being calculated twice and may increase runtime."
        )

  def calculate_epoch_and_sample_count(self, current_step):
    """Calculates the steps per epoch, current epoch number,
    and the number of samples seen by the model.

    Parameters
    ----------
    current_step: int
      the training step of the model

    """
    self.steps_per_epoch = math.ceil(self.train_dataset_size /
                                     self.wandb_run.config.batch_size)
    self.epoch_num = current_step / self.steps_per_epoch
    self.sample_count = current_step * self.wandb_run.config.batch_size

  def log(self, model, extra_data, step):
    """Logs the metrics and other extra data to W&B.

    Parameters
    ----------
    model: tf.keras.Model
     the Keras model implementing the calculation
    extra_data: dict
     extra data to be logged alongside calculated metrics
    step: int
     the step number
    """

    all_data = dict({})
    all_data.update(extra_data)
    all_data.update({
        'train/epoch': self.epoch_num,
        'train/sample_count': self.sample_count
    })

    if self.metrics is not None and self.metrics:
      # Get Training Metrics (interval dependent)
      if self.logging_strategy == "step" and step % self.wandb_run.config.log_frequency == 0:
        scores = model.evaluate(self.datasets["train"], self.metrics)
        scores = {'train/' + k: v for k, v in scores.items()}
        all_data.update(scores)
      elif self.logging_strategy == "epoch" and step % self.steps_per_epoch == 0:
        scores = model.evaluate(self.datasets["train"], self.metrics)
        scores = {'train/' + k: v for k, v in scores.items()}
        all_data.update(scores)

      # Get Eval Metrics (interval dependent)
      if self.datasets["eval"] is not None:
        if self.logging_strategy == "step" and step % self.wandb_run.config.log_frequency == 0:
          scores = model.evaluate(self.datasets["eval"], self.metrics)
          scores = {'eval/' + k: v for k, v in scores.items()}
          all_data.update(scores)
        elif self.logging_strategy == "epoch" and step % self.steps_per_epoch == 0:
          scores = model.evaluate(self.datasets["eval"], self.metrics)
          scores = {'eval/' + k: v for k, v in scores.items()}
          all_data.update(scores)

    self.wandb_run.log(all_data, step=step)

  def finish(self):
    """Finishes and closes the W&B run.
    Save run history data as field if configured to do that.
    """
    if self.save_run_history:
      self.run_history = copy.deepcopy(self.wandb_run.history)
    self.wandb_run.finish()

  def update_config(self, config_data):
    """Updates the W&B configuration.
    Parameters
    ----------
    config_data: dict
      additional configuration data to add
    """
    self.wandb_run.config.update(config_data)
