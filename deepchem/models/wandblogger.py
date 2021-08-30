import logging
import importlib.util
from typing import Optional, Union, List, Dict

logger = logging.getLogger(__name__)


def is_wandb_available():
  return importlib.util.find_spec("wandb") is not None


class WandbLogger(object):
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

    self.save_run_history = save_run_history

    # set wandb init arguments
    self.wandb_init_params = dict(
        name=name,
        project=project,
        entity=entity,
        mode=mode,
        id=id,
        resume=resume,
        anonymous=anonymous)
    self.wandb_init_params.update(**kwargs)
    self.initialized = False

    # Dataset ids are used to differentiate datasets seen by the logger
    self.dataset_ids: List[Union[int, str]] = []

  def setup(self):
    """Initializes a W&B run and create a run object.
    If a pre-existing run is already initialized, use that instead.
    """
    if self._wandb.run is None:
      self.wandb_run = self._wandb.init(**self.wandb_init_params)
    else:
      self.wandb_run = self._wandb.run
    self.initialized = True

  def log_data(self,
               data: Dict,
               step: int,
               dataset_id: Optional[Union[int, str]] = None):
    """Log data to W&B.

    Parameters
    ----------
    data: dict
      the data to be logged to W&B
    step: int
      the step number at which the data is to be logged
    dataset_id: int or str, optional (default None)
      the unique id of the dataset to differentiate during logging.
      Typically used when there are multiple ValidationCallbacks with
      different datasets.
    """
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

    # log data
    self.wandb_run.log(data, step=step)

  def finish(self):
    """Finishes and closes the W&B run.
    Save run history data as field if configured to do that.
    """
    if self.save_run_history:
      history = self.wandb_run.history._data
      self.run_history = history
    if self.wandb_run is not None:
      self.wandb_run.finish()

  def update_config(self, config_data):
    """Updates the W&B configuration.
    Parameters
    ----------
    config_data: dict
      additional configuration data to add
    """
    self.wandb_run.config.update(config_data)
