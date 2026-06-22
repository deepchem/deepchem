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
                 tokenizer_path: str = 'seyonec/PubChem10M_SMILES_BPE_60k',
                 n_tasks: int = 1,
                 config: Dict[Any, Any] = {},
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        model: PreTrainedModel
        chemberta_config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                                         **config)
        if task == 'mlm':
            model = BertForMaskedLM(chemberta_config) # More changes
        elif task == 'mtr':
            chemberta_config.problem_type = 'regression'
            chemberta_config.num_labels = n_tasks
            model = BertForSequenceClassification(chemberta_config)
        elif task == 'regression':
            chemberta_config.problem_type = 'regression'
            chemberta_config.num_labels = n_tasks
            model = BertForSequenceClassification(chemberta_config)
        elif task == 'classification':
            if n_tasks == 1:
                chemberta_config.problem_type = 'single_label_classification'
            else:
                chemberta_config.problem_type = 'multi_label_classification'
                chemberta_config.num_labels = n_tasks
            model = BertForSequenceClassification(chemberta_config)
        else:
            raise ValueError('invalid task specification')

        super(Chemberta, self).__init__(model=model,
                                        task=task,
                                        tokenizer=tokenizer,
                                        **kwargs)
