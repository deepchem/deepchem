from typing import Any, Tuple
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
    """DNABERT-2 Model

    DNABERT-2 is a transformer style model for learning on DNA sequences.
    The model architecture is based on a modified BERT architecture (MosaicBERT)
    with ALiBi (Attention with Linear Biases) positional encoding and Flash
    Attention support. The model can be used for both pretraining an embedding
    and finetuning for downstream applications.

    The model supports two types of pretraining tasks - pretraining via masked language
    modeling and pretraining via multi-task regression. To pretrain via masked language
    modeling task, use task = `mlm` and for pretraining via multitask regression task,
    use task = `mtr`. The model supports the regression, classification and multitask
    regression finetuning tasks and they can be specified using `regression`, `classification`
    and `mtr` as arguments to the `task` keyword during model initialisation.

    The model uses a tokenizer to create input tokens from raw DNA sequences.
    The default tokenizer is a Byte-Pair Encoding tokenizer trained on multi-species
    genomes and loaded from HuggingFace model hub
    (https://huggingface.co/zhihan1996/DNABERT-2-117M).


    Parameters
    ----------
    task: str
        The task defines the type of learning task in the model. The supported tasks are
         - `mlm` - masked language modeling commonly used in pretraining
         - `mtr` - multitask regression - a task used for both pretraining base models and finetuning
         - `regression` - use it for regression tasks, like property prediction
         - `classification` - use it for classification tasks
    tokenizer_path: str
        Path containing pretrained tokenizer used to tokenize DNA sequences for model inputs. The tokenizer path can either be a HuggingFace tokenizer model or a path in the local machine containing the tokenizer.
    n_tasks: int, default 1
        Number of prediction targets for a multitask learning model

    Example
    -------
    >>> import os
    >>> import tempfile
    >>> import shutil
    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> import pandas as pd
    >>> import deepchem as dc
    >>> sequences = ["ATGCGTACGTTAGCTAGC", "GGCTAACCGTATCGGATC"]
    >>> labels = [3.112, 2.432]
    >>> df = pd.DataFrame(list(zip(sequences, labels)), columns=["sequences", "task1"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="sequences", featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)

    >>> # pretraining
    >>> from deepchem.models.torch_models.dnabert import Dnabert
    >>> pretrain_model_dir = os.path.join(tempdir, 'pretrain-model')
    >>> tokenizer_path = "IronHead44/DNABERT-2-117M"
    >>> pretrain_model = Dnabert(task='mlm', model_dir=pretrain_model_dir, tokenizer_path=tokenizer_path)
    >>> pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)

    >>> # finetuning in regression mode
    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> finetune_model = Dnabert(task='regression', model_dir=finetune_model_dir, tokenizer_path=tokenizer_path)
    >>> finetune_model.load_from_pretrained(pretrain_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))

    >>> # removing temporary directory
    >>> if os.path.exists(tempdir):
    ...     shutil.rmtree(tempdir)


    Reference
    ---------
    .. Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., & Liu, H. (2024). DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. arXiv preprint arXiv:2306.15006.
    """

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
            model = AutoModelForMaskedLM.from_pretrained(tokenizer_path,
                                                         config=dnabert_config,
                                                         trust_remote_code=True)
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
        """
        Prepares a batch of data for the model based on the specified task. It overrides the _prepare_batch
        of parent class for the following condition:-

        - When n_task == 1 and task == 'classification', CrossEntropyLoss is used which takes input in
        long int format.
        - When n_task > 1 and task == 'classification', BCEWithLogitsLoss is used which takes input in
        float format.
        """
        genomic_batch, y, w = batch
        tokens = self.tokenizer(genomic_batch[0].tolist(),
                                padding=True,
                                return_tensors="pt")

        if self.task == 'mlm':
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device),
            }
            return inputs, None, w

        elif self.task in ["regression", "classification", "mtr"]:
            if y is not None:
                # y is None during predict
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
