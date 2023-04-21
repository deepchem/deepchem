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
        elif self.task == 'pretraining':
            # We use CrossEntropyLoss for pretraining as it was the default loss for many 
            # cases in the transformers library
            loss_fn = nn.CrossEntropyLoss()
        super().__init__(model, loss_fn)

    def load_from_pretrained(self, path: str):
        if isinstance(str, path):
            self.model.model.load_from_pretrained(path)
        # TODO Load from deepchem model checkpoint 

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        if self.task == 'pretraining':
            # FIXME we are assuming here that the pretraining task is masked language modeling
            # and hence tokenizing with masking
            smiles_batch, y, w = batch
            masked_smiles_tokens = self._mask_smiles_string(smiles_batch)
            # TODO How to handle masked labels as output in huggingface?
            # How does grover handle it?
            return masked_smiles_tokens, y, w
        elif self.task == 'finetuning':
            # tokenize without masking
            # TODO Should this be a batch of tensors?
            smiles_tokens = []
            for smiles_str in smiles_batch:
                tokens = self.tokenizer(smiles_str)
                smiles_tokens.append(tokens)
            return smiles_tokens, y, w

    def _mask_smiles_string(self, smiles_batch):
        # Use a tokenizer to batch smiles data and do random masking
        masked_smiles_tokens = []
        for smiles_str in smiles_batch:
            tokens = self.tokenizer(smiles_str)
            # do a random masking
            for i, token in enumerate(tokens):
                if np.random.binomial(1, 0.15):
                    # with 0.15 probability mask
                    if np.random.binomial(1, 0.8):
                        # Q: Will all huggingface tokenizers have mask_token_id? 
                        tokens[i] = tokenizer.mask_token_id
                    elif np.random.binomial(1, 0.5):
                        # if not chosen for masking, we either replace with random token id
                        # with probability of 0.1 or leave it unchanged
                        tokens[i] = np.random.randint(tokenizer.n_tokens)
                    else:
                        # leave it unchanged
                        pass
            masked_smiles_tokens.append(tokens)
        return masked_smiles_tokens 
