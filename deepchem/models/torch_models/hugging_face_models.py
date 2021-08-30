"""
Hugging Face Model Wrappers
"""
from typing import Optional
import warnings
try:
  from transformers import BertModel, BertConfig
except ModuleNotFoundError:
  raise ImportError(
      "DeepChem's wrappers for HuggingFace transformers require transformers.")

from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import Loss, L2Loss


# TODO write new examples
class BertModelWrapper(TorchModel):
  """Wrapper for the HuggingFace BertModel, loosely based on dc.attentivefp

  Notes
  -----
  This class requires transformers to be installed.
  """

  def __init__(self,
               model: BertModel,
               loss: Loss = L2Loss(),
               config: Optional[BertConfig] = None,
               **kwargs):
    """
    Parameters
    ----------
    model: BertModel
      The HuggingFace PreTrainedModel object to be used for training.
    loss: Loss
      A dc.models.lossess.Loss object used to calculate losses.
    config: Optional[PreTrainedConfig]
      If config is not None, `model` will be **reinitialized** with the config
      object provided.
      If config is None, `model` will be used as-is.
    kwargs
      Any additional TorchModel keyword arguments you would like to pass.
    """
    if config is None:
      self.model = model
    else:
      self.model = model.__init__(config=config)
    super().__init__(model, loss, **kwargs)

  def get_num_tasks(self):
    warnings.warn("get_num_tasks is not implemented in the BertModel class")

  def get_task_type(self):
    warnings.warn("get_task_type is not implemented in the BertModel class")

  def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)
