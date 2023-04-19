from typing import TYPE_CHECKING
import torch.nn as nn
import torch.functional as F
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

class HuggingFaceModel(TorchModel):
    """HuggingFace model wrapper 

    The class provides a wrapper for wrapping models from the `HuggingFace
    ecosystem in DeepChem and training it via DeepChem's api.

    Parameters
    ----------
    models: transformers.modeling_utils.PreTrainedModel
        The HuggingFace model to wrap.
    task: str
        Pretraining or finetuning task
    mode: str, optional (default None)
        The mode in which the model is being used. If `regression`, the model
    is used with a regression head attached. If `classification`, the model is used with
    a classification head attached.
    n_tasks: int, optional (default None)
        The number of tasks for the model to predict. This is only used if
    task is finetuning.
    """
    def __init__(self, model: 'PreTrainedModel', task: str, mode: Optional[str] = None, n_tasks: Optional[int] = None):
        
        if self.task == 'finetuning':
            assert self.mode is not None, 'Specify mode for finetuning task'
            if self.mode == 'regression':
                head = nn.Linear(in_features=model.config.hidden_size, out_features=n_tasks)
                loss_fn = L2Loss()
            elif self.mode == 'classification':
                head = nn.Linear(in_features=model.config.hidden_size, out_features=n_tasks * 2)
                loss_fn = nn.BCEWithLogitsLoss()
            model = nn.Sequential(model, head)
        
        super().__init__(model, loss_fn)

    def load_from_pretrained(self, path: str):
        if isinstance(str, path):
            self.model.model.load_from_pretrained(path)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        pass
