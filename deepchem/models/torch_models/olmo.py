from typing import Dict, Any, Tuple
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import GPTNeoXTokenizerFast, OlmoForCausalLM, OlmoConfig
from transformers.modeling_utils import PreTrainedModel
try:
    import torch
    has_torch = True
except:
    has_torch = False


class Olmo(HuggingFaceModel):
    """Olmo model for molecular representation learning."""
    def __init__(self,
                 task: str,
                 tokenizer_path: str = 'allenai/OLMo-7B-hf',
                 n_tasks: int = 1,
                 config: Dict[Any, Any] = {},
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_path)
        model: PreTrainedModel
        olmo_config = OlmoConfig(vocab_size=tokenizer.vocab_size,
                                         **config)
        if task == 'clm':
            model = OlmoForCausalLM(olmo_config)
        super(Olmo, self).__init__(model=model,
                                        task=task,
                                        tokenizer=tokenizer,
                                        **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """
        Prepares a batch of data for the model based on the specified task. It overrides the _prepare_batch
        of parent class for the following condition:-

        - When n_task == 1 and task == 'classification', CrossEntropyLoss is used which takes input in
        long int format.
        - When n_task > 1 and task == 'classification', BCEWithLogitsLoss is used which takes input in
        float format.
        """

        smiles_batch, y, w = batch
        tokens = self.tokenizer(smiles_batch[0].tolist(),
                                 padding=True,
                                return_tensors="pt")

        if self.task == 'clm':
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device),
            }
            return inputs, None, w
