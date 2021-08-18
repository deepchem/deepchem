"""
Hugging Face Model Wrappers
"""
from typing import Optional
import warnings
try:
  from transformers import PreTrainedModel, PretrainedConfig
except ModuleNotFoundError:
  raise ImportError("HuggingFace wrappers cannot function without transformers.")

from deepchem.models.torch_models.torch_model import TorchModel


# TODO write new examples
class HuggingFacePTModel(TorchModel):
  """Wrapper for HuggingFace Models, loosely based on dc.attentivefp

  Notes
  -----
  This class requires transformers to be installed.
  """

  def __init__(self,
               model: PreTrainedModel,
               config: Optional[PretrainedConfig] = None,
               **kwargs):
    """
    Parameters
    ----------
    model: PreTrainedModel
      The HuggingFace PreTrainedModel object to be used for training.
    config: Optional[PreTrainedConfig]
      If config is not None, `model` will be **reinitialized** with the config
      object provided.
      If config is None, `model` will be used as-is.
    kwargs
      Any additional TorchModel keyword arguments you would like to pass.
    """
    if isinstance(config, None):
      self.model = model
    else:
      self.model = model.__init__(config=config)
    super().__init__(model, **kwargs)

  def get_num_tasks(self):
    warnings.warn("get_num_tasks is not implemented in HuggingFacePTModel")

  def get_task_type(self):
    warnings.warn("get_task_type is not implemented in HuggingFacePTModel")
