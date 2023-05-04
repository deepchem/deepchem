from typing import Optional
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification)
from transformers.models.roberta.tokenization_roberta_fast import \
    RobertaTokenizerFast


class Chemberta(HuggingFaceModel):

    def __init__(self,
                 task: str,
                 mode: Optional[str] = None,
                 tokenizer_path: str = 'seyonec/PubChem10M_SMILES_BPE_60k',
                 **kwargs):
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        if task == 'pretraining':
            config = RobertaConfig(vocab_size=tokenizer.vocab_size)
            model = RobertaForMaskedLM(config)
        elif task == 'finetuning':
            assert mode, 'specify finetuning mode - classification or regression'
            if mode == 'regression':
                config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                                       problem_type='regression',
                                       num_labels=1)
                model = RobertaForSequenceClassification(config)
            elif mode == 'classification':
                config = RobertaConfig(vocab_size=tokenizer.vocab_size)
                model = RobertaForSequenceClassification(config)
            else:
                raise ValueError('invalid mode specification')
        else:
            raise ValueError('invalid task specification')

        super(Chemberta, self).__init__(model=model,
                                        task=task,
                                        tokenizer=tokenizer,
                                        **kwargs)
