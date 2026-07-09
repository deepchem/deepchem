from typing import Dict, Any, Tuple
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers.modeling_utils import PreTrainedModel
try:
    import torch
    has_torch = True
except:
    has_torch = False


class Dnabert(HuggingFaceModel):


    def __init__(self,
                 task: str,
                 tokenizer_path: str = 'IronHead44/DNABERT-2-117M',
                 n_tasks: int = 1,
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                  trust_remote_code=True)
        model: PreTrainedModel
        dnabert_config = AutoConfig.from_pretrained(tokenizer_path,
                                                    trust_remote_code=True)
        dnabert_config.pad_token_id = tokenizer.pad_token_id
        dnabert_config.is_decoder = False
        if task == 'mlm':
            model = AutoModelForMaskedLM.from_pretrained(
                tokenizer_path, config=dnabert_config, trust_remote_code=True)
        elif task == 'mtr':
            dnabert_config.problem_type = 'regression'
            dnabert_config.num_labels = n_tasks
            model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path, config=dnabert_config, trust_remote_code=True)
        elif task == 'regression':
            dnabert_config.problem_type = 'regression'
            dnabert_config.num_labels = n_tasks
            model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path, config=dnabert_config, trust_remote_code=True)
        elif task == 'classification':
            if n_tasks == 1:
                dnabert_config.problem_type = 'single_label_classification'
            else:
                dnabert_config.problem_type = 'multi_label_classification'
                dnabert_config.num_labels = n_tasks
            model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path, config=dnabert_config, trust_remote_code=True)
        else:
            raise ValueError('invalid task specification')

        super(Dnabert, self).__init__(model=model,
                                     task=task,
                                     tokenizer=tokenizer,
                                     **kwargs)
    

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        genomic_batch, y, w = batch
        tokens = self.tokenizer(genomic_batch[0].tolist(),
                    padding=True,
                    return_tensors="pt")

        if self.task == 'mlm':
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids']
            )
            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device),
            }
            return inputs, None, w
        
        elif self.task in ["regression", "classification", "mtr"]:
            if y is not None:
                y = torch.from_numpy(y[0])
                if self.task == 'regression' or self.task == 'mtr':
                    y = y.float().to(self.device)
                elif self.task == 'classification':
                    if self.n_tasks == 1:
                        y = y.long().to(self.device)
                    else:
                        y = y.float().to(self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)
            
            inputs = {**tokens, 'labels': y}
            return inputs, y, w