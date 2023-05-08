from typing import Optional
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification)
from transformers.models.roberta.tokenization_roberta_fast import \
    RobertaTokenizerFast


class Chemberta(HuggingFaceModel):
    """Chemberta Model

    Chemberta is a transformer style model for learning on SMILES strings.
    The model architecture is based on the RoBERTa architecture with a masked
    language modeling task for pretraining. The pretrained model can be finetuned for
    downstream applications in regression and classification mode.

    The model uses a tokenizer To create input tokens for the models from the SMILES strings.
    The default tokenizer model is a byte-pair encoding tokenizer trained on PubChem10M dataset
    and loaded from huggingFace model hub (https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_60k).


    Parameters
    ----------
    task: str
        Pretraining or finetuning task
    mode: str, default None
        regression mode or finetuning mode. Mode is None for pretraining task.
    tokenizer_path: str
        Path containing pretrained tokenizer used to tokenize SMILES string for model inputs. The tokenizer path can either be a huggingFace tokenizer model or a path in the local machine containing the tokenizer.

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
    >>> pretrain_model = Chemberta(task='pretraining', model_dir=pretrain_model_dir, tokenizer_path=tokenizer_path)
    >>> pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)

    >>> # finetuning in regression mode
    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> finetune_model = Chemberta(task='finetuning', model_dir=finetune_model_dir, tokenizer_path=tokenizer_path, mode='regression')
    >>> finetune_model.load_from_pretrained(pretrain_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))


    Reference
    ---------
    .. Ahmad, Walid, et al. "Chemberta-2: Towards chemical foundation models." arXiv preprint arXiv:2209.01712 (2022).
    """

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
