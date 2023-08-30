from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification)
from transformers.models.roberta.tokenization_roberta_fast import \
    RobertaTokenizerFast
from transformers.modeling_utils import PreTrainedModel


class Chemberta(HuggingFaceModel):
    """Chemberta Model

    Chemberta is a transformer style model for learning on SMILES strings.
    The model architecture is based on the RoBERTa architecture. The model
    has can be used for both pretraining an embedding and finetuning for
    downstream applications.

    The model supports two types of pretraining tasks - pretraining via masked language
    modeling and pretraining via multi-task regression. To pretrain via masked language
    modeling task, use task = `mlm` and for pretraining via multitask regression task,
    use task = `mtr`. The model supports the regression, classification and multitask
    regression finetuning tasks and they can be specified using `regression`, `classification`
    and `mtr` as arguments to the `task` keyword during model initialisation.

    The model uses a tokenizer To create input tokens for the models from the SMILES strings.
    The default tokenizer model is a byte-pair encoding tokenizer trained on PubChem10M dataset
    and loaded from huggingFace model hub (https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_60k).


    Parameters
    ----------
    task: str
        The task defines the type of learning task in the model. The supported tasks are
         - `mlm` - masked language modeling commonly used in pretraining
         - `mtr` - multitask regression - a task used for both pretraining base models and finetuning
         - `regression` - use it for regression tasks, like property prediction
         - `classification` - use it for classification tasks
    tokenizer_path: str
        Path containing pretrained tokenizer used to tokenize SMILES string for model inputs. The tokenizer path can either be a huggingFace tokenizer model or a path in the local machine containing the tokenizer.
    n_tasks: int, default 1
        Number of prediction targets for a multitask learning model

    Example
    -------
    >>> import os
    >>> import tempfile
    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> import pandas as pd
    >>> import deepchem as dc
    >>> smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    >>> labels = [3.112,2.432]
    >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="smiles", featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)

    >>> # pretraining
    >>> from deepchem.models.torch_models.chemberta import Chemberta
    >>> pretrain_model_dir = os.path.join(tempdir, 'pretrain-model')
    >>> tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"
    >>> pretrain_model = Chemberta(task='mlm', model_dir=pretrain_model_dir, tokenizer_path=tokenizer_path)  # mlm pretraining
    >>> pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)

    >>> # finetuning in regression mode
    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> finetune_model = Chemberta(task='regression', model_dir=finetune_model_dir, tokenizer_path=tokenizer_path)
    >>> finetune_model.load_from_pretrained(pretrain_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))


    Reference
    ---------
    .. Chithrananda, S., Grand, G., & Ramsundar, B. (2020). Chemberta: Large-scale self-supervised pretraining for molecular property prediction. arXiv preprint arXiv:2010.09885.
    .. Ahmad, Walid, et al. "Chemberta-2: Towards chemical foundation models." arXiv preprint arXiv:2209.01712 (2022).
    """

    def __init__(self,
                 task: str,
                 tokenizer_path: str = 'seyonec/PubChem10M_SMILES_BPE_60k',
                 n_tasks: int = 1,
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        model: PreTrainedModel
        chemberta_config = RobertaConfig(vocab_size=tokenizer.vocab_size)
        if task == 'mlm':
            model = RobertaForMaskedLM(chemberta_config)
        elif task == 'mtr':
            chemberta_config.problem_type = 'regression'
            chemberta_config.num_labels = n_tasks
            model = RobertaForSequenceClassification(chemberta_config)
        elif task == 'regression':
            chemberta_config.problem_type = 'regression'
            chemberta_config.num_labels = n_tasks
            model = RobertaForSequenceClassification(chemberta_config)
        elif task == 'classification':
            if n_tasks == 1:
                chemberta_config.problem_type = 'single_label_classification'
            else:
                chemberta_config.problem_type = 'multi_label_classification'
            model = RobertaForSequenceClassification(chemberta_config)
        else:
            raise ValueError('invalid task specification')

        super(Chemberta, self).__init__(model=model,
                                        task=task,
                                        tokenizer=tokenizer,
                                        **kwargs)
