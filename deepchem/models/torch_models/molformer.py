from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, AutoModelForSequenceClassification
try:
    import torch
    has_torch = True
except:
    has_torch = False


class MoLFormer(HuggingFaceModel):
    """
    MoLFormer is a large-scale chemical language model designed with the intention of learning a model trained
    on small molecules which are represented as SMILES strings. MoLFormer leverges masked language modeling
    and employs a linear attention Transformer combined with rotary embeddings.

    MoLFormer-XL-both-10pct is the version trained on 10% ZINC + 10% PubChem.

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
    >>> from deepchem.models.torch_models.molformer import MoLFormer
    >>> pretrain_model_dir = os.path.join(tempdir, 'pretrain-molformer-model')
    >>> tokenizer_path = "ibm/MoLFormer-XL-both-10pct"
    >>> pretrain_model = MoLFormer(task='mlm', model_dir=pretrain_model_dir, tokenizer_path=tokenizer_path)  # mlm pretraining
    >>> pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)

    >>> # finetuning in regression mode
    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> finetune_model = MoLFormer(task='regression', model_dir=finetune_model_dir, tokenizer_path=tokenizer_path)
    >>> finetune_model.load_from_pretrained(pretrain_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))


    Reference
    ---------
    .. Ross, Jerret and Belgodere, Brian and Chenthamarakshan, Vijil and Padhi, Inkit and Mroueh, Youssef and
    .. Das, Payel "Large-Scale Chemical Language Representations Capture Molecular Structure and Properties"
    .. https://doi.org/10.48550/arxiv.2106.09553
    """

    def __init__(self,
                 task: str,
                 tokenizer_path: str = 'ibm/MoLFormer-XL-both-10pct',
                 n_tasks: int = 1,
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                  trust_remote_code=True)
        molformer_config = AutoConfig.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        if task == 'mlm':
            model = AutoModelForMaskedLM.from_config(molformer_config,
                                                     trust_remote_code=True)
        elif task == 'mtr':
            molformer_config.problem_type = 'regression'
            molformer_config.num_labels = n_tasks
            model = AutoModelForSequenceClassification.from_config(
                config=molformer_config, trust_remote_code=True)
        elif task == 'regression':
            molformer_config.problem_type = 'regression'
            molformer_config.num_labels = n_tasks
            model = AutoModelForSequenceClassification.from_config(
                config=molformer_config, trust_remote_code=True)
        elif task == 'classification':
            if n_tasks == 1:
                molformer_config.problem_type = 'single_label_classification'
            else:
                molformer_config.num_labels = n_tasks
                molformer_config.problem_type = 'multi_label_classification'
            model = AutoModelForSequenceClassification.from_config(
                molformer_config, trust_remote_code=True)
        else:
            raise ValueError('invalid task specification')

        super(MoLFormer, self).__init__(model=model,
                                        task=task,
                                        tokenizer=tokenizer,
                                        **kwargs)

    def _prepare_batch(self, batch):
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

        if self.task == 'mlm':
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device),
            }
            return inputs, None, w
        elif self.task in ['regression', 'classification', 'mtr']:
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
