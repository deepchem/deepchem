import logging
import time
from collections.abc import Sequence as SequenceCollection
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional,
                    Tuple, Union, Dict)

import numpy as np
import torch
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models import TorchModel
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.typing import LossFn, OneOrMany
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto import (AutoModel, AutoModelForCausalLM,
                                       AutoModelForSequenceClassification,
                                       AutoModelForMaskedLM,
                                       AutoModelForUniversalSegmentation,
                                       AutoTokenizer)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import transformers
    from transformers.modeling_utils import PreTrainedModel


class HuggingFaceModel(TorchModel):
    r"""Wrapper class that wraps HuggingFace models as DeepChem models

    The class provides a wrapper for wrapping models from HuggingFace
    ecosystem in DeepChem and training it via DeepChem's api. The reason
    for this might be that you might want to do an apples-to-apples comparison
    between HuggingFace from the transformers library and DeepChem library.

    The `HuggingFaceModel` has a Has-A relationship by wrapping models from
    `transformers` library. Once a model is wrapped, DeepChem's API are used
    for training, prediction, evaluation and other downstream tasks.

    A `HuggingFaceModel` wrapper also has a `tokenizer` which tokenizes raw
    SMILES strings into tokens to be used by downstream models.  The SMILES
    strings are generally stored in the `X` attribute of deepchem.data.Dataset object'.
    This differs from the DeepChem standard workflow as tokenization is done
    on the fly here. The approach allows us to leverage `transformers` library's fast
    tokenization algorithms and other utilities like data collation, random masking of tokens
    for masked language model training etc.


    Parameters
    ----------
    model: transformers.modeling_utils.PreTrainedModel
        The HuggingFace model to wrap.
    task: str, (optional, default None)
        The task defines the type of learning task in the model. The supported tasks are
         - `mlm` - masked language modeling commonly used in pretraining
         - `mtr` - multitask regression - a task used for both pretraining base models and finetuning
         - `regression` - use it for regression tasks, like property prediction
         - `classification` - use it for classification tasks

        When the task is not specified or None, the wrapper returns raw output of the HuggingFaceModel.
        In cases where the HuggingFaceModel is a model without a task specific head, this output will be
        the last hidden states.
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
        Tokenizer
    config: dict, (optional, default None)
        A dictionary of model configuration parameters that will be passed to the Hugging Face
        `AutoModel` classes via `**kwargs` when loading from the hf_checkpoint. These parameters
        are typically used to customize the behavior and architecture of the underlying transformer
        model (e.g., number of layers, hidden size, dropout rates, etc.). When loading from pretrained
        from hf_checkpoint, If any keys in `config` match configuration attributes supported by
        the specific Hugging Face `AutoModel` being used, they will override the default settings
        for that model.

    Example
    -------
    >>> import os
    >>> import tempfile
    >>> import shutil
    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> smiles = ['CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1', 'CC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1', \
    ...     'COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC', 'OCCn1cc(CNc2cccc3c2CCCC3)nn1', \
    ...     'CCCCCCc1ccc(C#Cc2ccc(C#CC3=CC=C(CCC)CC3)c(C3CCCCC3)c2)c(F)c1', 'nO=C(NCc1ccc(F)cc1)N1CC=C(c2c[nH]c3ccccc23)CC1']
    >>> filepath = os.path.join(tempdir, 'smiles.txt')
    >>> f = open(filepath, 'w')
    >>> f.write('\n'.join(smiles))
    253
    >>> f.close()

    >>> # preparing tokenizer
    >>> from tokenizers import ByteLevelBPETokenizer
    >>> from transformers.models.roberta import RobertaTokenizerFast
    >>> tokenizer = ByteLevelBPETokenizer()
    >>> tokenizer.train(files=filepath, vocab_size=1_000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    >>> tokenizer_path = os.path.join(tempdir, 'tokenizer')
    >>> os.makedirs(tokenizer_path)
    >>> result = tokenizer.save_model(tokenizer_path)
    >>> tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

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
    >>> from deepchem.models.torch_models.hf_models import HuggingFaceModel
    >>> from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
    >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
    >>> model = RobertaForMaskedLM(config)
    >>> hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='mlm', model_dir='model-dir')
    >>> training_loss = hf_model.fit(dataset, nb_epoch=1)

    >>> # finetuning a regression model
    >>> from transformers.models.roberta import RobertaForSequenceClassification
    >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size, problem_type='regression', num_labels=1)
    >>> model = RobertaForSequenceClassification(config)
    >>> hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='regression', model_dir='model-dir')
    >>> hf_model.load_from_pretrained()
    >>> training_loss = hf_model.fit(dataset, nb_epoch=1)
    >>> prediction = hf_model.predict(dataset)  # prediction
    >>> eval_results = hf_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))

    >>> # finetune a classification model
    >>> # making dataset suitable for classification
    >>> import numpy as np
    >>> y = np.random.choice([0, 1], size=dataset.y.shape)
    >>> dataset = dc.data.NumpyDataset(X=dataset.X, y=y, w=dataset.w, ids=dataset.ids)

    >>> from transformers import RobertaForSequenceClassification
    >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
    >>> model = RobertaForSequenceClassification(config)
    >>> hf_model = HuggingFaceModel(model=model, task='classification', tokenizer=tokenizer)
    >>> training_loss = hf_model.fit(dataset, nb_epoch=1)
    >>> predictions = hf_model.predict(dataset)
    >>> eval_result = hf_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.f1_score))

    >>> # removing temporary directory
    >>> if os.path.exists(tempdir):
    ...     shutil.rmtree(tempdir)
    """

    def __init__(
            self,
            model: 'PreTrainedModel',
            tokenizer: 'transformers.tokenization_utils.PreTrainedTokenizer',
            task: Optional[str] = None,
            config: Optional[Dict] = None,
            **kwargs):
        self.task = task
        self.tokenizer = tokenizer
        if self.task == 'mlm':
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer)
        else:
            self.data_collator = None  # type: ignore
        # Ignoring type. For TorchModel, loss is a required argument but HuggingFace computes
        # loss during the forward iteration, removing the need for a loss function.
        if config:
            self.config = config
        else:
            self.config = {}
        super(HuggingFaceModel, self).__init__(
            model=model,
            loss=None,  # type: ignore
            **kwargs)

    def load_from_pretrained(  # type: ignore
            self,
            model_dir: Optional[str] = None,
            from_hf_checkpoint: bool = False):
        """Load HuggingFace model from a pretrained checkpoint.

        The utility can be used for loading a model from a checkpoint.
        Given `model_dir`, it checks for existing checkpoint in the directory.
        If a checkpoint exists, the models state is loaded from the checkpoint.

        If the option `from_hf_checkpoint` is set as True, then it loads a pretrained
        model using HuggingFace models `from_pretrained` method. This option
        interprets model_dir as a model id of a pretrained model hosted inside a model repo
        on huggingface.co or path to directory containing model weights saved using `save_pretrained`
        method of a HuggingFace model.

        Parameter
        ----------
        model_dir: str
            Directory containing model checkpoint
        from_hf_checkpoint: bool, default False
            Loads a pretrained model from HuggingFace checkpoint.

        Example
        -------
        >>> from transformers import RobertaTokenizerFast
        >>> tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_60k")

        >>> from deepchem.models.torch_models.hf_models import HuggingFaceModel
        >>> from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
        >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
        >>> model = RobertaForMaskedLM(config)
        >>> pretrain_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='mlm', model_dir='model-dir')
        >>> pretrain_model.save_checkpoint()

        >>> from transformers import RobertaForSequenceClassification
        >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
        >>> model = RobertaForSequenceClassification(config)
        >>> finetune_model = HuggingFaceModel(model=model, task='classification', tokenizer=tokenizer, model_dir='model-dir')

        >>> finetune_model.load_from_pretrained()

        Note
        ----
        1. Use `load_from_pretrained` method only to load a pretrained model - a
            model trained on a different task like Masked Language Modeling or
            Multitask Regression. To `restore` a model, use the `restore` method.

        2. A pretrain model has different number of target tasks for pretraining and a finetune
            model has different number of target tasks for finetuning. Thus, they both have different
            number of projection outputs in the last layer. To avoid a mismatch
            in the weights of the output projection layer (last layer) between
            the pretrain model and current model, we delete the projection
            layers weights.
        """
        if model_dir is None:
            model_dir = self.model_dir

        if from_hf_checkpoint:
            # FIXME Transformers library has an api like AutoModel.from_pretrained. It allows to
            # initialise and create a model instance directly without requiring a class instance initialisation step.
            # To use `load_from_pretrained` in DeepChem, we need to follow a two step process
            # of initialising class instance and then loading weights via `load_from_pretrained`.
            if self.task == 'mlm':
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_dir, trust_remote_code=True, **self.config)
            elif self.task in ['mtr', 'regression', 'classification']:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, trust_remote_code=True, **self.config)
            elif self.task == "universal_segmentation":
                self.model = AutoModelForUniversalSegmentation.from_pretrained(
                    model_dir, trust_remote_code=True, **self.config)
            else:
                self.model = AutoModel.from_pretrained(model_dir,
                                                       trust_remote_code=True,
                                                       **self.config)
        elif not from_hf_checkpoint:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            else:
                checkpoint = checkpoints[0]
                data = torch.load(checkpoint, map_location=self.device)
                # Delete keys of output projection layer (last layer) as the number of
                # tasks (projections) in pretrain model and the current model
                # might vary.

                # When using Distributed Data Parallel (DDP) for training models, PyTorch automatically
                # wraps model parameters in a module. prefix. This can cause issues when loading or
                # saving model states because the key names in state_dict differ from their original
                # single-GPU counterparts. To address this, model_state_dict is updated by removing
                # the "module." prefix when saving or loading models.

                data['model_state_dict'] = {
                    key.replace("module.", ""): value
                    for key, value in data['model_state_dict'].items()
                }
                keys = data['model_state_dict'].keys()
                if 'classifier.out_proj.weight' in keys:
                    del data['model_state_dict']['classifier.out_proj.weight']
                if 'classifier.out_proj.bias' in keys:
                    del data['model_state_dict']['classifier.out_proj.bias']
                if 'classifier.dense.bias' in keys:
                    del data['model_state_dict']['classifier.dense.bias']
                if 'classifier.dense.weight' in keys:
                    del data['model_state_dict']['classifier.dense.weight']
                self.model.load_state_dict(data['model_state_dict'],
                                           strict=False)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
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
                    y = y.long().to(self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, 'labels': y}
            return inputs, y, w

    def fit_generator(self,
                      generator: Iterable[Tuple[Any, Any, Any]],
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False,
                      variables: Optional[Union[List[torch.nn.Parameter],
                                                torch.nn.ParameterList]] = None,
                      loss: Optional[LossFn] = None,
                      callbacks: Union[Callable, List[Callable]] = [],
                      all_losses: Optional[List[float]] = None) -> float:
        """Train this model on data from a generator.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: list of torch.nn.Parameter
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step, **kwargs) that will be invoked
            after every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.

        Returns
        -------
        The average loss over the most recent checkpoint interval

        Note
        ----
        A HuggingFace model can return embeddings (last hidden state), attentions.
        Support must be added to return the embeddings to the user, so that it can
        be used for other downstream applications.
        """
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate,
                              LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        # Main training loop.

        for batch in generator:
            if restore:
                self.restore()
                restore = False
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self._prepare_batch(batch)

            optimizer.zero_grad()
            outputs = self.model(**inputs)

            batch_loss = outputs.get("loss")
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step

            avg_loss += batch_loss

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info('Ending global_step %d: Average loss %g' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                try:
                    # NOTE In DeepChem > 2.8.0, callback signature is updated to allow
                    # variable arguments.
                    c(self, current_step, iteration_loss=batch_loss)
                except TypeError:
                    # DeepChem <= 2.8.0, the callback should have this signature.
                    c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        if checkpoint_interval > 0:
            self.save_checkpoint(max_checkpoints_to_keep)

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
                 transformers: List[Transformer], uncertainty: bool,
                 other_output_types: Optional[OneOrMany[str]]):
        """Predicts output for data provided by generator.

        This is the private implementation of prediction. Do not
        call it directly. Instead call one of the public prediction methods.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        transformers: list of dc.trans.Transformers
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        uncertainty: bool
            specifies whether this is being called as part of estimating uncertainty.
            If True, it sets the training flag so that dropout will be enabled, and
            returns the values of the uncertainty outputs.
        other_output_types: list, optional
            Provides a list of other output_types (strings) to predict from model.

        Returns
        -------
            a NumPy array of the model produces a single output, or a list of arrays
            if it produces multiple outputs

        Note
        ----
        A HuggingFace model does not output uncertainity. The argument is here
        since it is also present in TorchModel. Similarly, other variables like
        other_output_types are also not used. Instead, a HuggingFace model outputs
        loss, logits, hidden state and attentions.
        """
        results: Optional[List[List[np.ndarray]]] = None
        variances: Optional[List[List[np.ndarray]]] = None
        if uncertainty and (other_output_types is not None):
            raise ValueError(
                'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
            )
        if uncertainty:
            if self._variance_outputs is None or len(
                    self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs'
                )
        if other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )
        self._ensure_built()
        self.model.eval()
        for batch in generator:
            inputs, labels, weights = batch
            inputs, _, _ = self._prepare_batch((inputs, None, None))

            # Invoke the model.
            output_values = self.model(**inputs)
            output_values = output_values.get('logits')

            if isinstance(output_values, torch.Tensor):
                output_values = [output_values]
            output_values = [t.detach().cpu().numpy() for t in output_values]
            # Apply tranformers and record results.
            if uncertainty:
                var = [output_values[i] for i in self._variance_outputs]
                if variances is None:
                    variances = [var]
                else:
                    for i, t in enumerate(var):
                        variances[i].append(t)
            access_values = []
            if other_output_types:
                access_values += self._other_outputs
            elif self._prediction_outputs is not None:
                access_values += self._prediction_outputs

            if len(access_values) > 0:
                output_values = [output_values[i] for i in access_values]

            if len(transformers) > 0:
                if len(output_values) > 1:
                    raise ValueError(
                        "predict() does not support Transformers for models with multiple outputs."
                    )
                elif len(output_values) == 1:
                    output_values = [
                        undo_transforms(output_values[0], transformers)
                    ]
            if results is None:
                results = [[] for i in range(len(output_values))]
            for i, t in enumerate(output_values):
                results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        final_variances = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))

        if uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances)

        if len(final_results) == 1:
            return final_results[0]
        else:
            return np.array(final_results)

    def fill_mask(self,
                  inputs: Union[str, List[str]],
                  top_k: int = 5) -> Union[List[Dict], List[List[Dict]]]:
        """Implements the HuggingFace 'fill_mask' pipeline from HuggingFace.
        https://huggingface.co/docs/transformers/main_classes/pipelines

        Takes as input a sequence or list of sequences where each sequence
        containts a single masked position and returns a list of dictionaries per sequence
        containing the filled sequence, the token, and the score for that token.

        Parameters
        ----------
        inputs : Union[str, List[str]]
            One or several texts (or one list of texts) with masked tokens.
        top_k : int, optional
            The number of predictions to return for each mask. Default is 5.

        Returns
        -------
        Union[List[Dict], List[List[Dict]]]
            A list or a list of list of dictionaries with the following keys:
            - sequence (str): The corresponding input with the mask token prediction.
            - score (float): The corresponding probability.
            - token (int): The predicted token id (to replace the masked one).
            - token_str (str): The predicted token (to replace the masked one)
        """

        # First make sure tha the model is successfully loaded, then set to eval mode.
        self._ensure_built()
        self.model.eval()

        # Ensure that the inputs are made into a list of len() >= 1.
        if isinstance(inputs, str):
            inputs = [inputs]

        results = []
        # Iterate over the input sequences (NOTE: DO NOT Parallelize)
        for text in inputs:
            encoded_input = self.tokenizer(text,
                                           return_tensors='pt').to(self.device)
            # Find all the occurrences where the mask token idx is used
            mask_token_index = torch.where(
                encoded_input["input_ids"] == self.tokenizer.mask_token_id)[1]
            # Ensure that the masked token index appears EXACTLY once.
            assert mask_token_index.numel(
            ) == 1, f"Sequence has masked indices at: {list(mask_token_index)}. Please ensure that only one position is masked in the sequence."

            with torch.no_grad():
                output = self.model(**encoded_input)

            # Grab the logits and take distribution at the masked token idx
            # Then take the top_k indices (which correspond to the token)
            logits = output.logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_k_tokens = torch.topk(mask_token_logits, top_k,
                                      dim=1).indices[0].tolist()

            # Decode the sequence with each of the top_k tokens inserted
            # Calculate the score as the probability of that token in the sequence.
            text_results = []
            for token in top_k_tokens:
                token_str = self.tokenizer.decode([token])
                filled_text = text.replace(self.tokenizer.mask_token, token_str)
                score = torch.softmax(mask_token_logits, dim=1)[0, token].item()
                text_results.append({
                    'sequence': filled_text,
                    'score': score,
                    'token': token,
                    'token_str': token_str
                })
            results.append(text_results)

        return results[0] if len(results) == 1 else results


class OLMoModel(HuggingFaceModel):
    """DeepChem wrapper for AllenAI OLMo-7B (decoder-only causal LM).

    Wraps ``allenai/OLMo-7B`` (or any compatible OLMo checkpoint) inside
    DeepChem's ``HuggingFaceModel`` abstraction.  For the GSoC scope this
    implementation focuses on **inference / generation** only.  The model
    receives one or more SMILES strings (or free-form molecular text) and
    returns the generated continuation produced by OLMo's causal language
    modelling head.

    Why a subclass and not just ``HuggingFaceModel`` directly?
    -----------------------------------------------------------
    * ``HuggingFaceModel._prepare_batch`` has no ``generation`` branch.
      Causal generation needs *no* label tensor — only ``input_ids`` and
      ``attention_mask``.
    * ``HuggingFaceModel._predict`` fetches ``logits`` from the model
      output, which is irrelevant for generation; we need ``model.generate()``.
    * ``HuggingFaceModel.load_from_pretrained`` has no
      ``AutoModelForCausalLM`` branch, so loading from a HF checkpoint
      would fall through to the bare ``AutoModel`` path (no LM head).

    Unsloth support
    ---------------
    `Unsloth <https://github.com/unslothai/unsloth>`_ is an optional
    acceleration library that rewrites key transformer kernels (attention,
    RoPE, cross-entropy) in Triton, giving 2-5× faster inference and
    training while using significantly less VRAM through 4-bit quantisation.
    It is entirely optional — the model loads and runs correctly without it.

    To enable unsloth, pass ``use_unsloth=True``.  The library must already
    be installed (``pip install unsloth``).  A default configuration is
    provided via ``DEFAULT_UNSLOTH_CONFIG``; override individual keys by
    passing a partial dict to ``unsloth_config``.

    Parameters
    ----------
    hf_model_name_or_path : str, optional
        HuggingFace model id or local path.  Defaults to
        ``"allenai/OLMo-7B"``.
    max_length : int, optional
        Maximum number of *new* tokens to generate per prompt.  Default 128.
    generation_kwargs : dict, optional
        Extra keyword arguments forwarded verbatim to ``model.generate()``.
        Common keys: ``temperature``, ``top_p``, ``top_k``,
        ``repetition_penalty``, ``do_sample``, ``num_beams``.
    use_unsloth : bool, optional
        If ``True``, load the model via Unsloth's ``FastLanguageModel`` for
        optimised inference.  Requires the ``unsloth`` package.
        Defaults to ``False``.
    unsloth_config : dict, optional
        Overrides for the default unsloth configuration.  Only the keys you
        supply are changed; everything else stays at the defaults defined in
        ``OLMoModel.DEFAULT_UNSLOTH_CONFIG``.  Ignored when
        ``use_unsloth=False``.

        Supported keys and their defaults:

        .. code-block:: python

            {
                "max_seq_length": 2048,   # context window passed to unsloth
                "dtype": None,            # None = auto-detect (bf16 on Ampere+)
                "load_in_4bit": True,     # 4-bit NF4 quantisation
            }

    **kwargs
        Remaining arguments are forwarded to ``HuggingFaceModel.__init__``
        and ultimately to ``TorchModel``.

    Examples
    --------
    Standard loading (no unsloth):

    >>> model = OLMoModel(hf_model_name_or_path="allenai/OLMo-7B")
    >>> outputs = model.generate(["CC(=O)Oc1ccccc1C(=O)O is a molecule that"])
    >>> print(outputs[0])

    Unsloth with default config (4-bit, auto dtype):

    >>> model = OLMoModel(use_unsloth=True)
    >>> outputs = model.generate(["CCO has the IUPAC name"])

    Unsloth with a custom config — keep 4-bit but use full float16 and a
    longer context window:

    >>> model = OLMoModel(
    ...     use_unsloth=True,
    ...     unsloth_config={"dtype": torch.float16, "max_seq_length": 4096},
    ... )
    >>> outputs = model.generate(["The solubility of caffeine is"])

    Unsloth with quantisation disabled (full weights, saves no VRAM but still
    uses fast kernels):

    >>> model = OLMoModel(
    ...     use_unsloth=True,
    ...     unsloth_config={"load_in_4bit": False},
    ... )
    """

    # ------------------------------------------------------------------
    # Default unsloth configuration
    # ------------------------------------------------------------------
    # These values are used whenever use_unsloth=True and the user has not
    # explicitly overridden a key via the unsloth_config argument.
    #
    # max_seq_length : maximum context window that unsloth pre-allocates
    #                  RoPE buffers for.  Longer = more VRAM.
    # dtype          : None lets unsloth auto-detect the best dtype for the
    #                  current GPU (bf16 on Ampere / fp16 elsewhere).
    # load_in_4bit   : enables NF4 quantisation via bitsandbytes, cutting
    #                  VRAM roughly in half versus fp16.
    DEFAULT_UNSLOTH_CONFIG: Dict = {
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": True,
    }

    # ------------------------------------------------------------------
    # Default LoRA configuration
    # ------------------------------------------------------------------
    # These values are passed to either Unsloth's FastLanguageModel.get_peft_model()
    # or directly to PEFT's LoraConfig when use_unsloth=False.
    #
    # r                        : LoRA rank. Higher = more trainable params, higher quality.
    #                            4/8/16/32/64 are common. 16 is a good starting point.
    # lora_alpha               : Scaling factor. Rule of thumb: set equal to r.
    # target_modules           : Which linear layers to attach LoRA to. These names
    #                            match OLMo-7B's HuggingFace module naming. Override
    #                            if you use a different OLMo variant.
    # lora_dropout             : Unsloth recommends 0.0 for speed. Use 0.05–0.1 if
    #                            you notice overfitting on small datasets.
    # bias                     : "none" is standard. "all" trains bias terms too.
    # use_gradient_checkpointing: "unsloth" enables Unsloth's smart gradient
    #                            checkpointing (30% less VRAM than standard).
    #                            Use True for standard HuggingFace checkpointing,
    #                            False to disable.
    # random_state             : Seed for LoRA weight initialisation.
    # use_rslora               : Rank-Stabilised LoRA — normalises by sqrt(r) instead
    #                            of r. Recommended for large ranks (r >= 32).
    # loftq_config             : LoftQ initialisation for better quantised training.
    #                            Pass a dict of LoftQ kwargs or None to disable.
    DEFAULT_LORA_CONFIG: Dict = {
        "r": 16,
        "lora_alpha": 16,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 42,
        "use_rslora": False,
        "loftq_config": None,
    }

    # ------------------------------------------------------------------
    # Default SFT (Supervised Fine-Tuning) training configuration
    # ------------------------------------------------------------------
    # These map 1-to-1 onto trl's SFTConfig (which wraps TrainingArguments).
    # Every key here can be overridden via the sft_config argument to sft().
    #
    # per_device_train_batch_size : Samples per GPU per step.  Keep at 1–4
    #                               for 7B on a 24 GB GPU.
    # gradient_accumulation_steps : Effective batch = batch_size × this value.
    #                               Use to simulate larger batches without OOM.
    # warmup_steps                : Linear warmup before full learning rate.
    # num_train_epochs            : Full passes over the dataset.  Set to None
    #                               and use max_steps for step-based training.
    # max_steps                   : Hard cap on training steps.  Overrides
    #                               num_train_epochs when set.
    # learning_rate               : Peak LR after warmup.  2e-4 is a solid
    #                               default for LoRA fine-tuning.
    # optim                       : "adamw_8bit" uses bitsandbytes 8-bit Adam,
    #                               halving optimiser memory vs full-precision.
    # weight_decay                : L2 regularisation on non-bias params.
    # lr_scheduler_type           : "linear" decays to 0 over training.
    #                               "cosine" often gives slightly better results.
    # fp16 / bf16                 : Set automatically based on GPU capability.
    #                               Override to force a specific dtype.
    # logging_steps               : Log loss every N steps.
    # save_steps                  : Save a checkpoint every N steps.
    # output_dir                  : Where SFTTrainer writes checkpoints.
    # dataset_text_field          : Column name in the HuggingFace dataset that
    #                               holds the training text.
    # max_seq_length              : Maximum token length per training sample.
    #                               Sequences are truncated to this.
    # packing                     : If True, SFTTrainer packs multiple short
    #                               sequences into one context window, greatly
    #                               improving throughput on short molecular texts.
    DEFAULT_SFT_CONFIG: Dict = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "num_train_epochs": 1,
        "max_steps": -1,
        "learning_rate": 2e-4,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "fp16": False,
        "bf16": False,
        "logging_steps": 10,
        "save_steps": 100,
        "output_dir": "olmo_sft_output",
        "dataset_text_field": "text",
        "max_seq_length": 512,
        "packing": True,
        "seed": 42,
    }

    def __init__(
            self,
            hf_model_name_or_path: str = "allenai/OLMo-7B",
            max_length: int = 128,
            generation_kwargs: Optional[Dict] = None,
            use_unsloth: bool = False,
            unsloth_config: Optional[Dict] = None,
            lora_config: Optional[Dict] = None,
            sft_config: Optional[Dict] = None,
            trust_remote_code: bool = True,
            **kwargs):

        self.hf_model_name_or_path = hf_model_name_or_path
        self.max_new_tokens = max_length
        self.generation_kwargs: Dict = generation_kwargs or {}
        self.use_unsloth: bool = use_unsloth
        self.trust_remote_code: bool = trust_remote_code

        # Build the effective unsloth config by layering user overrides on
        # top of the class-level defaults.  A shallow copy is important here
        # so that per-instance changes do not mutate DEFAULT_UNSLOTH_CONFIG.
        self.unsloth_config: Dict = {
            **OLMoModel.DEFAULT_UNSLOTH_CONFIG,
            **(unsloth_config or {}),
        }

        # Effective LoRA config — user overrides class defaults.
        # Stored at construction time so sft() can use it without requiring
        # the user to repeat the same dict on every call.
        self.lora_config: Dict = {
            **OLMoModel.DEFAULT_LORA_CONFIG,
            **(lora_config or {}),
        }

        # Effective SFT training config — same pattern.
        self.sft_config: Dict = {
            **OLMoModel.DEFAULT_SFT_CONFIG,
            **(sft_config or {}),
        }

        if use_unsloth:
            olmo, tokenizer = self._load_with_unsloth(hf_model_name_or_path)
        else:
            olmo, tokenizer = self._load_standard(hf_model_name_or_path)

        # OLMo has no pad token by default (decoder-only).
        # Reuse EOS as padding so the batch collation works correctly.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Pass task="generation" so the parent __init__ skips the
        # DataCollatorForLanguageModeling (which is only for masked LMs).
        super().__init__(
            model=olmo,
            tokenizer=tokenizer,
            task="generation",
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Private loading helpers
    # ------------------------------------------------------------------

    def _load_standard(self, path: str):
        """Load model and tokeniser via standard HuggingFace Transformers."""
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=self.trust_remote_code,
        )
        return model, tokenizer

    def _load_with_unsloth(self, path: str):
        """Load model and tokeniser via Unsloth's FastLanguageModel.

        Raises
        ------
        ImportError
            If the ``unsloth`` package is not installed.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ImportError(
                "Unsloth is not installed. "
                "Install it with:  pip install unsloth\n"
                "Or disable it by passing use_unsloth=False."
            ) from exc

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            trust_remote_code=self.trust_remote_code,
            **self.unsloth_config,
        )
        # Store the class so generate() can call for_inference() later.
        self._FastLanguageModel = FastLanguageModel
        return model, tokenizer

    # ------------------------------------------------------------------
    # Override 1: _prepare_batch
    # ------------------------------------------------------------------
    def _prepare_batch(
            self, batch: Tuple[Any, Any, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Any, Any]:
        """Prepare a single batch for the model.

        For the generation task we only need ``input_ids`` and
        ``attention_mask`` — no label tensor is required.  The parent
        implementation raises a ``KeyError`` for task="generation" because
        it has no branch for that value, so we override it here.

        For any other task recognised by the parent (``mlm``, ``regression``,
        ``classification``, ``mtr``) we delegate back to the parent so that
        OLMoModel remains usable for fine-tuning later.
        """
        if self.task != "generation":
            # Let the parent handle mlm / regression / classification / mtr.
            return super()._prepare_batch(batch)

        smiles_batch, y, w = batch

        # smiles_batch[0] is a numpy array of raw SMILES / prompt strings.
        prompts: List[str] = smiles_batch[0].tolist()

        tokens = self.tokenizer(
            prompts,
            padding=True,           # pad all sequences to the longest one
            truncation=True,        # truncate if a prompt is extremely long
            max_length=512,         # reasonable context window for inference
            return_tensors="pt",
        )

        # Move every tensor to the model's device (CPU / CUDA / MPS).
        inputs = {key: val.to(self.device) for key, val in tokens.items()}

        # y and w are unused during inference but kept for API consistency.
        return inputs, y, w

    # ------------------------------------------------------------------
    # Override 2: load_from_pretrained — add CausalLM branch
    # ------------------------------------------------------------------
    def load_from_pretrained(  # type: ignore[override]
            self,
            model_dir: Optional[str] = None,
            from_hf_checkpoint: bool = False) -> None:
        """Load OLMo weights from a checkpoint.

        Extends the parent with both an ``AutoModelForCausalLM`` branch and
        an Unsloth ``FastLanguageModel`` branch so that
        ``from_hf_checkpoint=True`` loads the correct model class regardless
        of whether unsloth is enabled.

        Parameters
        ----------
        model_dir : str, optional
            Path or HF hub id.  Falls back to ``self.hf_model_name_or_path``
            when not provided.
        from_hf_checkpoint : bool
            When ``True``, reload from a HuggingFace checkpoint using either
            the standard or unsloth loader depending on ``self.use_unsloth``.
            When ``False``, restore from a DeepChem ``torch.save`` checkpoint
            (same behaviour as the parent, unsloth has no effect here).
        """
        if from_hf_checkpoint:
            path = model_dir or self.hf_model_name_or_path
            if self.use_unsloth:
                # Re-use the same helper that __init__ calls so the
                # effective unsloth_config is applied consistently.
                self.model, self.tokenizer = self._load_with_unsloth(path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    trust_remote_code=self.trust_remote_code,
                    **self.config,
                )
        else:
            # Reuse the parent's checkpoint-restoration logic unchanged.
            super().load_from_pretrained(
                model_dir=model_dir,
                from_hf_checkpoint=False,
            )

    # ------------------------------------------------------------------
    # New public method: generate
    # ------------------------------------------------------------------
    def generate(
            self,
            inputs: Union[str, List[str]],
            max_new_tokens: Optional[int] = None,
            skip_special_tokens: bool = True,
            **decode_kwargs,
    ) -> List[str]:
        """Run autoregressive generation on one or more prompts.

        This is the primary inference entry-point for OLMoModel.  It
        tokenises the prompts, calls ``model.generate()``, and returns the
        decoded strings — stripping the original prompt so only the *newly
        generated* text is returned.

        When ``use_unsloth=True``, ``FastLanguageModel.for_inference()`` is
        called automatically before generation.  This patches the model's
        attention and FFN kernels in-place to use Unsloth's optimised Triton
        implementations — you do not need to call it yourself.

        Parameters
        ----------
        inputs : str or list of str
            One SMILES string / molecular prompt or a list of them.
        max_new_tokens : int, optional
            Override the instance-level ``max_new_tokens`` for this call.
        skip_special_tokens : bool
            If ``True`` (default), remove special tokens such as ``<eos>``
            from the decoded output.
        **decode_kwargs
            Any extra keyword arguments are merged with
            ``self.generation_kwargs`` and forwarded to ``model.generate()``.
            Call-level kwargs take priority over instance-level ones.

        Returns
        -------
        list of str
            One decoded string per input prompt, containing only the text
            generated *after* the prompt.

        Examples
        --------
        >>> model = OLMoModel()
        >>> model.generate(["The SMILES CC(=O)O represents"])
        ['acetic acid, also known as ethanoic acid.']
        """
        self._ensure_built()

        # When unsloth is active, swap to its optimised inference kernels.
        # for_inference() patches the model in-place and is idempotent, so
        # calling it on every generate() is safe (and cheap after the first
        # call).
        if self.use_unsloth:
            self._FastLanguageModel.for_inference(self.model)

        self.model.eval()

        # Normalise to list.
        if isinstance(inputs, str):
            inputs = [inputs]

        # Tokenise all prompts together for batched generation.
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.unsloth_config.get("max_seq_length", 512),
        )
        input_ids: torch.Tensor = encoded["input_ids"].to(self.device)
        attention_mask: torch.Tensor = encoded["attention_mask"].to(self.device)

        # Number of prompt tokens per sequence (needed to strip the prompt
        # from the output later).  All sequences are padded to the same
        # length, so we use the full width of input_ids as the prompt length.
        prompt_len: int = input_ids.shape[1]

        # Merge generation kwargs: instance defaults < call overrides.
        gen_kwargs = {**self.generation_kwargs, **decode_kwargs}
        gen_kwargs["max_new_tokens"] = max_new_tokens or self.max_new_tokens

        # OLMo uses EOS as the pad token; suppress pad-token warnings from
        # HuggingFace by explicitly setting pad_token_id.
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        with torch.no_grad():
            output_ids: torch.Tensor = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # output_ids includes the prompt tokens; slice them off so we return
        # only the newly generated part.
        new_ids = output_ids[:, prompt_len:]

        decoded: List[str] = self.tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return decoded

    # ------------------------------------------------------------------
    # Private helper: apply LoRA adapters
    # ------------------------------------------------------------------

    def _apply_lora_adapters(self, effective_lora: Dict) -> None:
        """Attach LoRA adapters to ``self.model`` in-place.

        Chooses between two backends depending on ``self.use_unsloth``:

        **Unsloth path** (``use_unsloth=True``):
            Calls ``FastLanguageModel.get_peft_model()``.  Unsloth's
            implementation fuses the LoRA adapters with its own Triton
            kernels, giving an additional 10–30 % speed boost on top of the
            base model optimisations.  It also enables Unsloth's "smart"
            gradient checkpointing which uses 30 % less VRAM than the
            standard HuggingFace version.

        **PEFT fallback** (``use_unsloth=False``):
            Calls ``peft.get_peft_model()`` with a ``LoraConfig`` built from
            the same effective config dict.  Produces identical weights; the
            only difference is speed and VRAM usage.

        Parameters
        ----------
        effective_lora : dict
            Fully-merged LoRA config (class defaults + ``__init__`` overrides
            + ``sft()`` call overrides).  Must contain at minimum the keys
            defined in ``DEFAULT_LORA_CONFIG``.

        Notes
        -----
        ``use_gradient_checkpointing`` and ``random_state`` are Unsloth-
        specific keys that are not recognised by PEFT's ``LoraConfig``.  They
        are stripped before constructing the PEFT config.
        """
        # Keys recognised by Unsloth but NOT by PEFT — remove before PEFT path.
        _unsloth_only_keys = {"use_gradient_checkpointing", "random_state"}

        if self.use_unsloth:
            if not hasattr(self, '_FastLanguageModel'):
                # _load_with_unsloth sets this; guard in case the model was
                # loaded via the standard path and unsloth was enabled later.
                try:
                    from unsloth import FastLanguageModel
                    self._FastLanguageModel = FastLanguageModel
                except ImportError as exc:
                    raise ImportError(
                        "Unsloth is not installed. "
                        "Install it with:  pip install unsloth"
                    ) from exc

            # Unsloth expects use_gradient_checkpointing as a direct kwarg.
            self.model = self._FastLanguageModel.get_peft_model(
                self.model,
                **effective_lora,
            )
            logger.info("LoRA adapters applied via Unsloth FastLanguageModel.")

        else:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError as exc:
                raise ImportError(
                    "PEFT is required for LoRA when use_unsloth=False. "
                    "Install it with:  pip install peft"
                ) from exc

            # Strip Unsloth-specific keys that PEFT does not understand.
            peft_lora = {
                k: v for k, v in effective_lora.items()
                if k not in _unsloth_only_keys
            }
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **peft_lora,
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()
            logger.info("LoRA adapters applied via PEFT.")

    # ------------------------------------------------------------------
    # Private helper: convert DeepChem dataset → HuggingFace Dataset
    # ------------------------------------------------------------------

    def _dc_dataset_to_hf(
            self,
            dataset,
            text_field: str = "text",
            formatting_func: Optional[Callable] = None,
    ):
        """Convert a DeepChem ``NumpyDataset`` to a HuggingFace ``Dataset``.

        The ``X`` attribute of a DeepChem dataset holds raw SMILES strings
        (or any text prompts) when a ``DummyFeaturizer`` was used at load
        time.  This helper turns those strings into a HuggingFace Dataset
        with a single text column suitable for ``SFTTrainer``.

        If a ``formatting_func`` is provided it is applied to each row to
        produce the final training string — useful for wrapping SMILES in an
        instruction template (e.g. Alpaca / ChatML format).

        Parameters
        ----------
        dataset : dc.data.Dataset or datasets.Dataset or list of str
            The training data.  If it is already a HuggingFace Dataset it is
            returned unchanged.  If it is a list of strings it is wrapped
            into a Dataset directly.
        text_field : str
            The column name to use for the text in the output Dataset.
        formatting_func : callable, optional
            A function of the form ``f(row: dict) -> str`` that takes a row
            dict (with at least a ``"text"`` key containing the raw SMILES)
            and returns a formatted training string.  Example::

                def format_row(row):
                    return (
                        f"### Molecule\\n{row['text']}\\n"
                        "### Description\\n"
                    )

        Returns
        -------
        datasets.Dataset
            A HuggingFace Dataset with a single column named ``text_field``.

        Raises
        ------
        ImportError
            If the ``datasets`` package is not installed.
        """
        try:
            from datasets import Dataset as HFDataset
        except ImportError as exc:
            raise ImportError(
                "The `datasets` package is required for SFT. "
                "Install with:  pip install datasets"
            ) from exc

        # Already a HuggingFace Dataset — nothing to do.
        if isinstance(dataset, HFDataset):
            return dataset

        # Plain list of strings.
        if isinstance(dataset, list):
            texts = dataset
        else:
            # DeepChem NumpyDataset: X is a numpy array where each element
            # is a SMILES string (when DummyFeaturizer was used).
            raw = dataset.X
            if raw.ndim == 2:
                # Shape (N, 1) — flatten to (N,)
                raw = raw[:, 0]
            texts = raw.tolist()

        if formatting_func is not None:
            texts = [formatting_func({text_field: t}) for t in texts]

        return HFDataset.from_dict({text_field: texts})

    # ------------------------------------------------------------------
    # Public method: sft — Supervised Fine-Tuning
    # ------------------------------------------------------------------

    def sft(
            self,
            dataset,
            lora_config: Optional[Dict] = None,
            sft_config: Optional[Dict] = None,
            formatting_func: Optional[Callable] = None,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ) -> None:
        """Fine-tune OLMoModel with Supervised Fine-Tuning (SFT) via LoRA.

        This is the primary training entry-point for the OLMoModel.  It
        attaches LoRA adapters to the frozen OLMo backbone, converts the
        input dataset to the format expected by ``trl.SFTTrainer``, and
        runs the training loop.  After ``sft()`` returns, ``self.model``
        holds the LoRA-adapted weights and ``generate()`` can be called
        immediately without any additional setup.

        Configuration precedence (highest wins):
            call-level ``lora_config`` / ``sft_config`` kwargs
                ↓
            instance-level ``self.lora_config`` / ``self.sft_config``
            (set via ``__init__`` arguments)
                ↓
            class-level ``DEFAULT_LORA_CONFIG`` / ``DEFAULT_SFT_CONFIG``

        Parameters
        ----------
        dataset : dc.data.Dataset, datasets.Dataset, or list of str
            Training data.  Accepted forms:

            * A DeepChem ``NumpyDataset`` whose ``X`` attribute contains
              SMILES strings loaded with ``DummyFeaturizer``.
            * A HuggingFace ``datasets.Dataset`` with a text column whose
              name matches ``sft_config["dataset_text_field"]`` (default
              ``"text"``).
            * A plain Python ``list`` of strings.

        lora_config : dict, optional
            Per-call overrides for LoRA configuration.  Merged on top of
            ``self.lora_config`` (which itself overrides class defaults).

            Full list of supported keys (with defaults from
            ``DEFAULT_LORA_CONFIG``):

            .. code-block:: python

                {
                    "r": 16,
                    "lora_alpha": 16,
                    "target_modules": [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    "lora_dropout": 0.0,
                    "bias": "none",
                    "use_gradient_checkpointing": "unsloth",
                    "random_state": 42,
                    "use_rslora": False,
                    "loftq_config": None,
                }

        sft_config : dict, optional
            Per-call overrides for SFT training configuration.  Merged on
            top of ``self.sft_config`` (which itself overrides class
            defaults).  All keys map directly to ``trl.SFTConfig`` / HF
            ``TrainingArguments``.

            Full list of supported keys (with defaults from
            ``DEFAULT_SFT_CONFIG``):

            .. code-block:: python

                {
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "warmup_steps": 5,
                    "num_train_epochs": 1,
                    "max_steps": -1,
                    "learning_rate": 2e-4,
                    "optim": "adamw_8bit",
                    "weight_decay": 0.01,
                    "lr_scheduler_type": "linear",
                    "fp16": False,
                    "bf16": False,
                    "logging_steps": 10,
                    "save_steps": 100,
                    "output_dir": "olmo_sft_output",
                    "dataset_text_field": "text",
                    "max_seq_length": 512,
                    "packing": True,
                    "seed": 42,
                }

        formatting_func : callable, optional
            A function ``f(row: dict) -> str`` applied to each training
            sample before it is fed to the tokeniser.  Use this to wrap
            SMILES strings in an instruction template, e.g.::

                def alpaca_format(row):
                    return (
                        "### Instruction:\\n"
                        "Describe the following molecule.\\n"
                        "### Input:\\n"
                        f"{row['text']}\\n"
                        "### Response:\\n"
                    )

            When ``formatting_func`` is provided the ``dataset_text_field``
            setting is ignored because SFTTrainer calls the function
            directly on each batch.

        resume_from_checkpoint : str or bool, optional
            Path to a checkpoint directory, or ``True`` to resume from the
            most recent checkpoint in ``output_dir``.  Forwarded verbatim to
            ``trainer.train()``.

        Returns
        -------
        None
            ``self.model`` is updated in-place with the LoRA-adapted weights.
            Call ``generate()`` directly after ``sft()`` — no extra setup
            required.

        Raises
        ------
        ImportError
            If ``trl`` is not installed (``pip install trl``).
        ImportError
            If ``datasets`` is not installed (``pip install datasets``).
        ImportError
            If ``peft`` is not installed and ``use_unsloth=False``
            (``pip install peft``).

        Examples
        --------
        Minimal fine-tune on a list of SMILES strings, all defaults:

        >>> model = OLMoModel(use_unsloth=True)
        >>> smiles = ["CC(=O)O", "CCO", "c1ccccc1", "CC(N)=O"]
        >>> model.sft(smiles)
        >>> outputs = model.generate(["The molecule CC(=O)O is"])

        Customise LoRA rank and SFT learning rate:

        >>> model = OLMoModel(use_unsloth=True)
        >>> model.sft(
        ...     dataset=smiles,
        ...     lora_config={"r": 32, "lora_alpha": 64, "use_rslora": True},
        ...     sft_config={"learning_rate": 5e-5, "num_train_epochs": 3},
        ... )

        Use an instruction template with a DeepChem dataset:

        >>> import deepchem as dc
        >>> loader = dc.data.CSVLoader(
        ...     ["label"], feature_field="smiles",
        ...     featurizer=dc.feat.DummyFeaturizer()
        ... )
        >>> dc_dataset = loader.create_dataset("molecules.csv")
        >>>
        >>> def format_smiles(row):
        ...     return (
        ...         "### Molecule\\n" + row["text"] +
        ...         "\\n### Description\\n"
        ...     )
        >>>
        >>> model = OLMoModel(use_unsloth=True)
        >>> model.sft(dc_dataset, formatting_func=format_smiles)

        Override LoRA targets to only fine-tune attention projections:

        >>> model.sft(
        ...     dataset=smiles,
        ...     lora_config={
        ...         "r": 8,
        ...         "target_modules": ["q_proj", "v_proj"],
        ...         "lora_dropout": 0.05,
        ...         "bias": "none",
        ...     },
        ...     sft_config={
        ...         "packing": False,
        ...         "max_seq_length": 256,
        ...         "output_dir": "my_lora_checkpoint",
        ...     },
        ... )

        Disable 4-bit quantisation and use full bf16:

        >>> model = OLMoModel(
        ...     use_unsloth=True,
        ...     unsloth_config={"load_in_4bit": False},
        ...     lora_config={"r": 64, "use_rslora": True},
        ...     sft_config={"bf16": True, "fp16": False},
        ... )
        >>> model.sft(smiles)
        """
        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError as exc:
            raise ImportError(
                "trl is required for SFT. "
                "Install it with:  pip install trl"
            ) from exc

        # ------------------------------------------------------------------
        # Step 1 — Resolve the three effective configs (precedence chain).
        # ------------------------------------------------------------------
        # LoRA: class defaults → instance (set in __init__) → call overrides
        effective_lora: Dict = {
            **OLMoModel.DEFAULT_LORA_CONFIG,
            **self.lora_config,
            **(lora_config or {}),
        }

        # SFT: class defaults → instance → call overrides
        effective_sft: Dict = {
            **OLMoModel.DEFAULT_SFT_CONFIG,
            **self.sft_config,
            **(sft_config or {}),
        }

        # Separate SFTTrainer-specific keys from TrainingArguments keys.
        # SFTConfig owns: dataset_text_field, max_seq_length, packing.
        # Everything else goes into TrainingArguments via SFTConfig's **kwargs.
        _sft_only_keys = {"dataset_text_field", "max_seq_length", "packing"}
        sft_only = {k: effective_sft.pop(k) for k in list(effective_sft)
                    if k in _sft_only_keys}

        dataset_text_field: str = sft_only.get("dataset_text_field", "text")
        max_seq_length: int = sft_only.get("max_seq_length", 512)
        packing: bool = sft_only.get("packing", True)

        logger.info("Effective LoRA config: %s", effective_lora)
        logger.info("Effective SFT training config: %s", effective_sft)

        # ------------------------------------------------------------------
        # Step 2 — Apply LoRA adapters to the frozen backbone.
        # ------------------------------------------------------------------
        self._apply_lora_adapters(effective_lora)

        # ------------------------------------------------------------------
        # Step 3 — Convert dataset to HuggingFace format.
        # ------------------------------------------------------------------
        hf_dataset = self._dc_dataset_to_hf(
            dataset,
            text_field=dataset_text_field,
            # Only apply formatting here if no formatting_func is given;
            # if formatting_func is given, SFTTrainer calls it directly.
            formatting_func=None,
        )

        # ------------------------------------------------------------------
        # Step 4 — Build SFTConfig (wraps TrainingArguments).
        # ------------------------------------------------------------------
        training_cfg = SFTConfig(
            max_seq_length=max_seq_length,
            packing=packing,
            dataset_text_field=dataset_text_field,
            **effective_sft,
        )

        # ------------------------------------------------------------------
        # Step 5 — Build SFTTrainer and run training.
        # ------------------------------------------------------------------
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=hf_dataset,
            args=training_cfg,
            # formatting_func is passed only when the user supplied one,
            # because SFTTrainer ignores dataset_text_field when a
            # formatting_func is present.
            **({"formatting_func": formatting_func}
               if formatting_func is not None else {}),
        )

        logger.info("Starting SFT training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("SFT training complete.")

        # Update self.model to the LoRA-adapted version returned by the
        # trainer so that generate() picks up the fine-tuned weights.
        self.model = trainer.model

        # Switch to optimised inference kernels if Unsloth is active.
        if self.use_unsloth and hasattr(self, '_FastLanguageModel'):
            self._FastLanguageModel.for_inference(self.model)
            logger.info("Switched to Unsloth inference mode.")