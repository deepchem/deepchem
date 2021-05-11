import torch
import tensorflow as tf
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.data import Dataset, NumpyDataset
from deepchem.metrics import Metric

class WandbLogger(object):
    """
  Weights & Biases Logger
  """

    def __init__(self,
                 datasets: List[Dataset],
                 metrics: List[Metric],
                 log_loss: bool = True,
                 name: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 offline: Optional[bool] = False,
                 id: Optional[str] = None,
                 anonymous: Optional[bool] = None,
                 version: Optional[str] = None,
                 project: Optional[str] = None,
                 log_model: Optional[bool] = False,
                 experiment=None,
                 prefix: Optional[str] = '',
                 **kwargs):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'You want to use `wandb` logger which is not installed yet,'
                ' install it with `pip install wandb`.'
            )

        if offline and log_model:
            # TODO: Different exception type?
            raise Exception(
                f'Providing log_model={log_model} and offline={offline} is an invalid configuration'
                ' since model checkpoints cannot be uploaded in offline mode.\n'
                'Hint: Set `offline=False` to log your model.'
            )
        self.base_model = None # will be set in KerasModel init
        self.datasets = datasets
        self.metrics = metrics
        self.log_loss = log_loss

        self.offline = offline
        self.log_model = log_model
        self.prefix = prefix
        self.experiment = experiment
        # set wandb init arguments
        anonymous_lut = {True: 'allow', False: None}
        self.wandb_init = dict(
            name=name,
            project=project,
            id=version or id,
            dir=save_dir,
            resume='allow',
            anonymous=anonymous_lut.get(anonymous, anonymous)
        )
        self.wandb_init.update(**kwargs)
        # extract parameters
        self.save_dir = self.wandb_init.get('dir')
        self.name = self.wandb_init.get('name')
        self.id = self.wandb_init.get('id')

        #Log the parameters of KerasModel

        self.wandb = wandb.init(**self.wandb_init) if wandb.run is None else wandb.run

    def save_model(self, model):
        #model is a tf.keras.Model
        return None

    def log_data(self, model, step):
        #model is a Deepchem KerasModel
        for dataset in self.datasets:
            scores = model.evaluate(dataset, self.metrics)
        self.wandb.log()


    def update_config(self):
        return None

