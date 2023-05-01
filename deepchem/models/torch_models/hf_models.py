import os
import logging
import time
import numpy as np
from collections.abc import Sequence as SequenceCollection
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional,
                    Tuple, Union)
import torch
from deepchem.trans import Transformer, undo_transforms
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models import TorchModel
from deepchem.utils.typing import LossFn, OneOrMany
from transformers.data.data_collator import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import transformers
    from transformers.modeling_utils import PreTrainedModel


class HuggingFaceModel(TorchModel):
    r"""HuggingFace model wrapper

    The class provides a wrapper for wrapping models from the `HuggingFace
    ecosystem in DeepChem and training it via DeepChem's api.

    Parameters
    ----------
    models: transformers.modeling_utils.PreTrainedModel
        The HuggingFace model to wrap.
    task: str
        Pretraining or finetuning task
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
        Tokenizer

   .. code-block:: python
         
        import os
        import tempfile
        tempdir = tempfile.mkdtemp()

        # preparing dataset
        filepath = os.path.join(tempdir, 'smiles.txt')
        with open(filepath, 'w') as f:
            f.write('CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1\nCC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1\n')
            f.write('COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC\nOCCn1cc(CNc2cccc3c2CCCC3)nn1\n')
            f.write('CCCCCCc1ccc(C#Cc2ccc(C#CC3=CC=C(CCC)CC3)c(C3CCCCC3)c2)c(F)c1\nO=C(NCc1ccc(F)cc1)N1CC=C(c2c[nH]c3ccccc23)CC1\n')

        # preparing tokenizer
        from tokenizers import ByteLevelBPETokenizer
        from transformers.models.roberta import RobertaTokenizerFast
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=filepath, vocab_size=1_000, min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
        tokenizer_path = os.path.join(tempdir, 'tokenizer')
        os.makedirs(tokenizer_path)
        tokenizer.save_model(tokenizer_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

        # preparing dataset
        import pandas as pd
        import deepchem as dc
        smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
        labels = [3.112,2.432]
        df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
        with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
            df.to_csv(tmpfile.name)
            loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
                                     featurizer=dc.feat.DummyFeaturizer())
            dataset = loader.create_dataset(tmpfile.name)

        # pretraining
        from deepchem.models.torch_models.hf_models import HuggingFaceModel
        from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
        config = RobertaConfig(vocab_size=tokenizer.vocab_size)
        model = RobertaForMaskedLM(config)
        hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='pretraining', model_dir='model-dir')
        hf_model.fit(dataset, nb_epoch=1)
 
        # finetuning a regression model
        from transformers.models.roberta import RobertaForSequenceClassification
        config = RobertaConfig(vocab_size=tokenizer.vocab_size, problem_type='regression', num_labels=1)
        model = RobertaForSequenceClassification(config)
        hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='finetuning', model_dir='model-dir')
        hf_model.load_from_pretrained()
        hf_model.fit(dataset, nb_epoch=1)
        hf_model.predict(dataset)  # prediction
        hf_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))  # evaluation

        # finetune a classification model
        # making dataset suitable for classification
        import numpy as np
        y = np.random.choice([0, 1], size=dataset.y.shape)
        dataset = dc.data.NumpyDataset(X=dataset.X, y=y, w=dataset.w, ids=dataset.ids) 

        from transformers import RobertaForSequenceClassification        
        config = RobertaConfig(vocab_size=tokenizer.vocab_size)
        model = RobertaForSequenceClassification(config)
        hf_model = HuggingFaceModel(model=model, task='finetuning', tokenizer=tokenizer) 
        hf_model.fit(dataset, nb_epoch=1)
        hf_model.predict(dataset)
        hf_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.f1_score))
    """

    def __init__(
            self, model: 'PreTrainedModel', task: str,
            tokenizer: 'transformers.tokenization_utils.PreTrainedTokenizer',
            **kwargs):
        self.task = task
        self.tokenizer = tokenizer
        if self.task == 'pretraining':
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer)
        # Ignoring type. For TorchModel, loss is a required argument but HuggingFace computes
        # loss during the forward iteration, removing the need for a loss function.
        super(HuggingFaceModel, self).__init__(
            model=model,
            loss=None,  # type: ignore
            **kwargs)

    def load_from_pretrained(self, path: str):  # type: ignore
        """Load HuggingFace mode from pretrained checkpoint

        Parameters
        ----------
        path: str
            Directory containing model checkpoint
        """
        self.model.model.load_from_pretrained(path)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        if self.task == 'pretraining':
            smiles_batch, _, w = batch
            tokens = self.tokenizer(smiles_batch[0].tolist(),
                                    padding=True,
                                    return_tensors="pt")
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {'input_ids': inputs, 'labels': labels}
            return inputs, None, w
        elif self.task == 'finetuning':
            smiles_batch, y, w = batch
            tokens = self.tokenizer(smiles_batch[0].tolist(),
                                    padding=True,
                                    return_tensors="pt")
            if y is not None:
                # y is None during predict
                y = torch.from_numpy(y[0])
                if self.model.config.problem_type == 'regression':
                    y = y.float()

            inputs = {**tokens, 'labels': y}
            return inputs, y, w

    def fit_generator(self,
                      generator: Iterable[Tuple[Any, Any, Any]],
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False,
                      variables: Optional[List[torch.nn.Parameter]] = None,
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
            one or more functions of the form f(model, step) that will be invoked after
            every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.

        Returns
        -------
        The average loss over the most recent checkpoint interval
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

            if self._loss_outputs is not None:
                outputs = [outputs[i] for i in self._loss_outputs]
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
            output_values = output_values.get('logits').detach().cpu().numpy()

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
