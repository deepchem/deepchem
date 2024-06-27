from transformers import BertForMaskedLM, BertForSequenceClassification, BertTokenizer, BertConfig
import torch.nn as nn
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from typing import Union


class ProtBERT(HuggingFaceModel):
    """
    ProtBERT model[1].

    ProtBERT model is based on the BERT architecture and the current implementation
    supports only MLM pretraining and classification mode, as described by the
    authors in HuggingFace[2]. The classification mode supports three tasks:
    membrane protein prediction, subcellular localization prediction, and custom
    classification tasks with a user-provided classifier head.

    The model converts the input protein sequence into a vector through a trained BERT tokenizer, which is then
    processed by the corresponding model based on the task. `BertForMaskedLM` is used to facilitate the MLM
    pretraining task. For the sequence classification task, we follow `BertForSequenceClassification` but change
    the classifier to a custom `nn.Module` if provided.


    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> import pandas as pd
    >>> import deepchem as dc
    >>> protein = ["MPCTTYLPLLLLLFLLPPPSVQSKV","SSGLFWMELLTQFVLTWPLVVIAFL"]
    >>> labels = [0,1]
    >>> df = pd.DataFrame(list(zip(protein, labels)), columns=["protein", "task1"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="protein", featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)

    >>> # pretraining
    >>> from deepchem.models.torch_models.prot_bert import ProtBERT
    >>> pretrain_model_dir = os.path.join(tempdir, 'pretrain-model')
    >>> model_path = 'Rostlab/prot_bert'
    >>> pretrain_model = ProtBERT(task='mlm', model_pretrain_dataset='uniref100', n_tasks=1, model_dir=pretrain_model_dir)  # mlm pretraining
    >>> pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)
    >>> del pretrain_model

    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> custom_torch_cls_seq_network = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Linear(512, 2))
    >>> finetune_model = ProtBERT(task='classification', model_pretrain_dataset='uniref100', n_tasks=1,cls_task="custom",cls_head=custom_torch_cls_seq_network, model_dir=finetune_model_dir)
    >>> finetune_model.load_from_pretrained(pretrain_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.accuracy_score))

    References
    ----------
    .. [1] Elnaggar, Ahmed, et al. "Prottrans: Toward understanding the language of life through self-supervised learning." IEEE transactions on pattern analysis and machine intelligence 44.10 (2021): 7112-7127.
    .. [2] https://huggingface.co/Rostlab/prot_bert

    """

    def __init__(self,
                 task: str,
                 model_pretrain_dataset: str = "uniref100",
                 n_tasks: int = 1,
                 cls_task: str = "custom",
                 cls_head: Union[nn.Module, None] = None,
                 n_classes: int = 2,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        task: str
            The task defines the type of learning task in the model. The supported tasks are:
            - `mlm` - masked language modeling commonly used in pretraining
            - `classification` - use it for classification tasks
        model_pretrain_dataset: str
            The dataset used for pretraining the model. Options are 'uniref100' and 'bfd'.
        n_tasks: int, default 1
            Number of prediction targets for a multitask learning model
        cls_task: str, default "custom"
            The specific classification task. Options are "membrane", "subcellular location", and "custom".
        cls_head: nn.Module, optional
            A custom classifier head to use for classification mode.
        n_classes: int, default 2
            Number of classes for classification.
        """

        self.n_tasks: int = n_tasks

        if model_pretrain_dataset == "uniref100":
            model_path = 'Rostlab/prot_bert'
        elif model_pretrain_dataset == "bfd":
            model_path = 'Rostlab/prot_bert_bfd'
        else:
            raise ValueError('Invalid pretraining dataset: {}.'.format(
                model_pretrain_dataset))

        tokenizer = BertTokenizer.from_pretrained(model_path,
                                                  do_lower_case=False)

        if task == "mlm":
            model = BertForMaskedLM.from_pretrained(model_path)
        elif task == "classification":
            if model_pretrain_dataset == "uniref100" and cls_task in [
                    "membrane", "subcellular location"
            ]:
                raise ValueError(
                    "Classification model for '{}' task is only available with BFD pretraining dataset."
                    .format(cls_task))

            if cls_task == "membrane":
                if cls_head is not None:
                    raise ValueError(
                        "Custom classifier head is not supported for 'membrane' task."
                    )
                model = BertForSequenceClassification.from_pretrained(
                    'Rostlab/prot_bert_bfd_membrane')

            elif cls_task == "subcellular location":
                if cls_head is not None:
                    raise ValueError(
                        "Custom classifier head is not supported for 'subcellular location' task."
                    )
                model = BertForSequenceClassification.from_pretrained(
                    'Rostlab/prot_bert_bfd_localization')

            elif cls_task == "custom":
                if isinstance(cls_head, nn.Module):
                    cls_net = cls_head
                else:
                    raise ValueError(
                        'Invalid classifier head type. Expected nn.Module but got {}.'
                        .format(type(cls_head)))

                protbert_config = BertConfig.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    vocab_size=tokenizer.vocab_size)
                protbert_config.num_labels = n_classes

                if n_tasks == 1:
                    protbert_config.problem_type = 'single_label_classification'
                else:
                    protbert_config.problem_type = 'multi_label_classification'

                model = BertForSequenceClassification.from_pretrained(
                    model_path, config=protbert_config)
                model.classifier = cls_net

            else:
                raise ValueError(
                    'Invalid classification task: {}. Expected "membrane", "subcellular location", or "custom".'
                    .format(cls_task))
        else:
            raise ValueError(
                'Invalid task specification: {}. Expected "mlm" or "classification".'
                .format(task))

        super().__init__(model=model, task=task, tokenizer=tokenizer, **kwargs)
