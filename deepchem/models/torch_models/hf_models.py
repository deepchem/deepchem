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
from transformers.models.auto import AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForUniversalSegmentation

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

    Example
    -------
    >>> import os
    >>> import tempfile
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
    """

    def __init__(
            self,
            model: 'PreTrainedModel',
            tokenizer: 'transformers.tokenization_utils.PreTrainedTokenizer',
            task: Optional[str] = None,
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
                self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
            elif self.task in ['mtr', 'regression', 'classification']:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir)
            elif self.task == "universal_segmentation":
                self.model = AutoModelForUniversalSegmentation.from_pretrained(
                    model_dir)
            else:
                self.model = AutoModel.from_pretrained(model_dir)
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
