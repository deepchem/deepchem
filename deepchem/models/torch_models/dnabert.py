from typing import Dict,Any,Tuple
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import (AutoConfig,AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer)
from transformers.modeling_utils import PreTrainedModel

try:
    import torch    
    has_torch=True
except:
    has_torch=False

class DNABERTModel(HuggingFaceModel):
    """DNABERT Model for DNA Sequenece analysis.
    
    DNABERT is a transformer-based model pretrained on genomic sequences.
    It can be used for both pretraining embeddings and fine tuning for downstream genomic applications
    such as promoter, sequence classification, splice site detection and sequenec rgeression. 
    
    The model supports multiple task types:
    -  `mlm`- Masked Language Modeling for pretraining.
    -  `regression`- Single or multi-task regression.
    -  `classification`- Single or multi-label classification.
    
    Parameters
    -----------
    task: str
        The learning task type. Supported tasks:
        - `mlm`- masked language modeling.
        -  `regression`- Regression Tasks (e.g- Binding Affinity Prediction)
        -  `classification`- Classification Tasks(e.g Promoter Detection)           # Reminder- Test all these

    model_name: str, optional(default "zhihan1996/DNABERT-2-117M")
        Hugging Face model identifier or local path
    
    n_tasks: int, default 1
        Number of prediction targets for a multitask learning model

    config : Dict[Any, Any], optional (default {})
        Additional configuration parameters for the model

    Example
    --------
    ### Need to fill- when testing is done

    Notes
    --------
    - DNABERT-2 uses k-mer tokenization optimized for DNA Sequences.
    - The model expects uppercase DNA Sequences (A,C,G,T).
    - For best results, sequences should be between 50-512 base pairs.

    References
    ----------
    .. [1] Zhou, Z., et al. "DNABERT-2: Efficient Foundation Model for 
       Multi-Species Genome." arXiv preprint arXiv:2306.15006 (2023).
    """

    def __init__(
            self,
            task:str,
            model_name:str='zhihan1996/DNABERT-2-117M',
            n_tasks:int=1,
            config:Dict[Any,Any]={},
            **kwargs
    ):
        self.n_tasks=n_tasks
        self.model_name=model_name
        tokenizer=AutoTokenizer.from_pretrained(
            model_name,
        )
        model_config=AutoConfig.from_pretrained(
            model_name,**config
        )
        model:PreTrainedModel
        if task=='mlm':
            model=AutoModelForMaskedLM.from_pretrained(
                model_name,
                config=model_config
            )
        elif task=='regression':
            model_config.problem_type='regression'
            model_config.num_labels=n_tasks
            model=AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=model_config,
            )
        elif task=='classification':
            if n_tasks==1:
                model_config.problem_type='single_label_classification'
                model_config.num_labels=2
            else:
                model_config.problem_type='multi_label_classification'
                model_config.num_labels=n_tasks

            model=AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=model_config,
            )
        else:
            raise ValueError('invalid task specification')
        
        super(DNABERTModel,self).__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            **kwargs
        )
    
    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Prepares a batch of DNA sequences for the model.

        Handles different label formats based on task type:
        - Classification (single-task): uses long int for CrossEntropyLoss
        - Classification (multi-task): uses float for BCEWithLogitsLoss
        - Regression: uses float
        - MLM: masks tokens for pretraining
        """
        sequences_batch, y, w = batch
        
        # Tokenize sequences
        tokens = self.tokenizer(
            sequences_batch[0].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        if self.task == 'mlm':
            # Masking tokens for pretraining
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids']
            )
            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device),
            }
            return inputs, None, w
        
        elif self.task in ['regression', 'classification']:
            if y is not None:
                # Convert labels to appropriate format
                y = torch.from_numpy(y[0])
                if self.task == 'regression':
                    y = y.float().to(self.device)
                elif self.task == 'classification':
                    if self.n_tasks == 1:
                        y = y.long().to(self.device)
                    else:
                        y = y.float().to(self.device)
            
            # Move tokens to device
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, 'labels': y}
            return inputs, y, w