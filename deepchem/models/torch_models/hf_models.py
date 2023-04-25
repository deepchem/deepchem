from typing import TYPE_CHECKING, Tuple, Any
import torch.nn as nn
import torch.functional as F
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import L2Loss

from transformers.data.data_collator import DataCollatorForLanguageModeling

if TYPE_CHECKING:
    import transformers
    from transformers.modeling_utils import PreTrainedModel


def hf_loss_fct(outputs, labels, weights):
    # Hacking around DeepChem's TorchModel.
    # In HuggingFace, the forward pass method also returns the loss - in the forward pass,
    # we pass in both inputs and labels, hence the forward pass can compute both predictions
    # and loss. So, we retrieve the loss attribute and return it.
    return outputs.get("loss")


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
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
        Tokenizer
    """

    def __init__(
            self, model: 'PreTrainedModel', task: str,
            tokenizer: 'transformers.tokenization_utils.PreTrainedTokenizer'):
        self.task = task
        self.tokenizer = tokenizer
        if self.task == 'pretraining':
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer)
        super(HuggingFaceModel, self).__init__(model=model, loss=hf_loss_fct)

    def load_from_pretrained(self, path: str):
        if isinstance(str, path):
            self.model.model.load_from_pretrained(path)
        # TODO Load from deepchem model checkpoint

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        if self.task == 'pretraining':
            smiles_batch, y, w = batch
            tokens = self.tokenizer(smiles_batch[0].tolist(),
                                    padding=True,
                                    return_tensors="pt")
            input_ids, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {'input_ids': input_ids, 'labels': labels}
            return inputs, labels, w
